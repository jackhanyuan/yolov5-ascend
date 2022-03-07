#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import struct
import acl
import os
from PIL import Image
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

class Net(object):
    def __init__(self, device_id, model_path):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.init_resource()

    def __del__(self):
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('Resources released successfully.')

    def init_resource(self):
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        # print("data interaction from host to device")
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        # print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        # print("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
       # print("data interaction from device to host success")
        result = self.get_result(res)
        # self._print_result(result)
        # tuple_st = struct.unpack("1000f", bytearray(result[0]))
        # vals = np.array(tuple_st).flatten()

        return result

    def run(self, images):
        acl.rt.set_context(self.context)
        self._data_from_host_to_device(images)
        self.forward()
        return self._data_from_device_to_host()

    def forward(self):
        # print('execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        # print('execute stage success')

    def _print_result(self, result):
        tuple_st = struct.unpack("1000f", bytearray(result[0]))
        vals = np.array(tuple_st).flatten()
        top_k = vals.argsort()[-1:-6:-1]
        print("======== top5 inference results: =============")

        for j in top_k:
            print("[%d]: %f" % (j, vals[j]))

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def get_result(self, output_data):
        dataset = []
        # print(output_data)
        for temp in output_data:
            size = temp["size"]
            ptr = temp["buffer"]
            data = acl.util.ptr_to_numpy(ptr, (size,), 1)
            dataset.append(data)
            # print(data)
        return dataset