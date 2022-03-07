#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# by [jackhanyuan](https://github.com/jackhanyuan) 07/03/2022

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import cv2
import acl
import torch
import numpy as np
from PIL import Image
from torchvision.ops import nms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from acl_net import Net
from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_ERROR_NONE, IMG_EXT

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def resize_img(input_img, target_size, padding=True):
    if padding:
        old_size = input_img.shape[0:2]
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        img_new = cv2.resize(input_img, (new_size[1], new_size[0]))
        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        resized_img = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    else:
        resized_img = cv2.resize(input_img, (target_size[1], target_size[0]))
    return resized_img


def load_label(label_name):
    label_lookup_path = label_name
    with open(label_lookup_path, 'r') as f:
        label_contents = f.readlines()

    labels = np.array(list(map(lambda x: x.strip(), label_contents)))
    return labels


def preprocess(img_data, input_shape=(320, 320), image_format='BGR', channel_first=False, mean=[0., 0., 0.],
               std=[255., 255, 255.], fp16=False, padding=True):
    image_file = Image.open(img_data)
    image_file = image_file.convert("RGB")
    org_img = np.array(image_file)
    # image_file = image_file.resize(input_shape)
    img = np.array(image_file)
    # rgb to bgr，改变通道顺序
    if image_format == 'BGR':
        org_img = org_img[:, :, ::-1]
        img = img[:, :, ::-1]
    img = resize_img(img, input_shape, padding)
    shape = img.shape
    if fp16:
        img = img.astype("float16")
    else:
        img = img.astype("float32")
    img[:, :, 0] -= mean[0]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[2]
    img[:, :, 0] /= std[0]
    img[:, :, 1] /= std[1]
    img[:, :, 2] /= std[2]
    img = img.reshape([1] + list(shape))
    if channel_first:
        img = img.transpose([0, 3, 1, 2])
    if fp16:
        img_bytes = np.frombuffer(img.tobytes(), np.float16)
    else:
        img_bytes = np.frombuffer(img.tobytes(), np.float32)
    return org_img, img, img_bytes


def draw_box(image, boxes, names, scores, show_label=True):
    image_h, image_w, _ = image.shape

    for i, box in enumerate(boxes):
        box = np.array(box[:4], dtype=np.int32)  # xyxy

        line_width = int(3)
        txt_color = (255, 255, 255)
        box_color = (58, 56, 255)

        p1, p2 = (box[0], box[1]), (box[2], box[3])
        image = cv2.rectangle(image, p1, p2, box_color, line_width)

        if show_label:
            tf = max(line_width - 1, 1)  # font thickness
            box_label = '%s: %.2f' % (names[i], scores[i])
            w, h = cv2.getTextSize(box_label, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            image = cv2.rectangle(image, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
            image = cv2.putText(image, box_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                                line_width / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return image
    
    
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'ascend/yolov5s.om')
    parser.add_argument('--labels', nargs='+', type=str, default=ROOT / 'ascend/yolov5.label')
    parser.add_argument('--imgsz', nargs='+', type=int, default=(640, 640), help='inference size h,w')
    parser.add_argument('--images-dir', type=str, default=ROOT / 'img')
    parser.add_argument('--output-dir', type=str, default=ROOT / 'img_out')
    parser.add_argument('--device', type=int, default=0, help='npu device id, i.e. 0 or 1')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--save-img', action='store_true', default=True, help='save image')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    t0 = time.perf_counter()
    print("ACL Init:")
    ret = acl.init()
    check_ret("acl.init", ret)
    device_id = opt.device

    # 1.Load model
    print("Loading model %s" % opt.weights)
    model_path = str(opt.weights)
    net = Net(device_id, model_path)

    # 2.Load label
    label_path = opt.labels
    labels = load_label(label_path)
    input_size = opt.imgsz

    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    agnostic_nms = opt.agnostic_nms
    max_det = opt.max_det
    fileter_classes = None
    
    # Directories
    output_dir = increment_path(Path(opt.output_dir) / 'exp', exist_ok=False)  # increment run
    (output_dir / 'labels' if opt.save_txt else output_dir).mkdir(parents=True, exist_ok=True)  # make dir


    # 3.Start Detect
    print()
    print("Start Detect:")
    
    images_dir = opt.images_dir
    images = sorted(os.listdir(images_dir))
    count = 0
    label_count = 0
    total_count = len(images)
    for image_name in images:
        t1 = time.perf_counter()
        count += 1

        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)

        org_img, image_npy, image_bytes = preprocess(image_path, input_shape=input_size, image_format='BGR', channel_first=True)
        result = net.run([image_bytes])
        pred = np.frombuffer(bytearray(result[0]), dtype=np.float32)
        
        # pred = pred.reshape(1, 102000, -1) # 1280 x 1280
        pred = pred.reshape(1, 25200, -1)  # 640 x 640

        # Apply NMS
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, fileter_classes, agnostic_nms, max_det=max_det)

        s = ""
        boxes = []
        annos = []
        names = []
        scores = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(input_size, det[:, :4], org_img.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    name = labels[c]
                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    score = float(conf)

                    boxes.append(box)
                    names.append(name)
                    scores.append(score)
                    annos.append(c)
                    # print("\t{}\t{:.3f}\t{} ".format(box, score, name))

                    if opt.save_txt:  # Write to file
                        txt_path = str(output_dir / 'labels' / os.path.splitext(image_name)[0])
                        line = cls, *xyxy, conf  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
        if opt.save_img:  # Add box to image
            out_img = org_img.copy()
            if len(boxes) > 0:
                label_count += 1
                out_img = draw_box(out_img, boxes, names, scores)
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, out_img)

        t2 = time.perf_counter()
        t  = t2 - t1
        print('image {}/{} {}: {}Done. ({:.3f}s)'.format(count, total_count, image_path, s, t))

    t3 = time.perf_counter()
    t  = int(t3 - t0)
    print('This detection cost {}s'.format(t))
    print("Results saved to %s" % output_dir)
    print("{} labels saved to {}".format(label_count, output_dir / 'labels')) if opt.save_txt else print()
    print()