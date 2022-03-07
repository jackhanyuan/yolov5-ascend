# YOLOv5 ascend 
This repo is YOLOv5 om model inference program on the Huawei Ascend platform.
All programs passed the test on Huawei Atlas 300I inference card (Ascend 310 AI CPU, CANN 5.0.2, npu-smi 21.0.2).
You can run demo by `python detect_yolov5_ascend.py`.

## Export om model 
(1) Training your YOLOv5 model by [ultralytics/yolov5](https://github.com/ultralytics/yolov5). Then export the pytorch model to onnx format.
```bash
# in yolov5 root path export pt model to onnx.
python export.py --weights yolov5s.pt --opset 12 --simplify --include onnx 
```

(2) On the Huawei Ascend platform, using the `atc` tool convert the onnx model to om model.
```bash
# in yolov5 root path export pt model to onnx.
atc --input_shape="images:1,3,640,640" --input_format=NCHW --output="yolov5s" --soc_version=Ascend310 --framework=5 --model="yolov5s.onnx" --output_type=FP32 
```

## Inference by Ascend NPU
(1)Clone repo and move `*.om model` to `yolov5-ascend/ascend/*.om`.
```bash
git clone https://github.com/jackhanyuan/yolov5-ascend
mv yolov5s.om yolov5-ascend/ascend/
```

(2)Edit label file in `yolov5-ascend/ascend/yolov5.label`.


(3)Run inference program.
```bash
python detect_yolov5_ascend.py
```
The result will save to `img_out` folder.