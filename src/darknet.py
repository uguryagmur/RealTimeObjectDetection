"""Darknet Neural Network class"""

from __future__ import division
from _io import TextIOWrapper

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from contextlib import contextmanager
try:
    from src.util import predict_transform
except ImportError:
    from util import predict_transform


class MaxPoolStride1(nn.Module):
    r"""
    A class represents a Max Pooling Layer with stride 1

    --------
    Attributes:
        kernel_size (int, list): kernel size of the max pooling layer

    --------
    Methods:
        forward(x) -> torch.Tensor:
            forward pass method of the layer class with input tensor x
    """

    def __init__(self, kernel_size):
        r"""Constructor of the Max-Pool strie=1 Class"""
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x) -> torch.Tensor:
        r"""Forward pass method for the layer class

        --------
        Arguments:
            x (torch.Tensor): input tensor for the layer
        """
        x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        x = nn.MaxPool2d(self.kernel_size, self.pad)(x)
        return x


class EmptyLayer(nn.Module):
    r"""Route Layer Class which is given in YOLO cfg files"""

    def __init__(self):
        r"""Constructor of the Empty Layer Class"""
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    r"""
    YOLO Detection Layer Class

    --------
    Attributes:
        anchors (list): anchor boxes list given in the YOLO cfg file
        CUDA (bool): CUDA flag to enable GPU usage

    --------
    Methods:
        forward(x, input_dim, num_classes, confidence) -> torch.Tensor:
            forward pass for detection layer function to detect the objects
            given in the classes with given confidence
    """

    def __init__(self, anchors, CUDA=False):
        r"""Constructor of the Detection Layer Class

        --------
        Arguments:
            anchors: anchor boxes list of the Darknet network
        """
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.CUDA = CUDA

    def forward(self, x, inp_dim, num_classes) -> torch.Tensor:
        r"""Forward pass definition of the Detection Layer

        --------
        Arguments:
            x (torch.Tensor): input tensor of the layer
            inp_dim (list): dimensions of the input file
            num_classes (int): number of the classes for object detection
        """
        x = x.data
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors,
                                       num_classes, CUDA=self.CUDA)
        return prediction


class Upsample(nn.Module):
    r"""Upsampling Layer Class

    --------
    Attributes:
        stride (int): stride of the upsampling layer (default=2)

    --------
    Methods:
        forward(x) -> torch.Tensor:
            forward pass of the Layer class for the given input tensor x
    """

    def __init__(self, stride=2):
        r"""Constructor of the Upsample Layer Class"""
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x) -> torch.Tensor:
        r"""Forward Pass of the Upsample Class

        --------
        Arguments:
            x (torch.Tensor): input tensor of the layer
        """
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = x.view(B, C, H, 1,
                   W, 1).expand(B, C, H, stride, W,
                                stride).contiguous().view(B, C, H*stride,
                                                          W*stride)
        return x


class Darknet(nn.Module):
    """YOLO Darknet Class

    --------
    Attributes:
        blocks (list): darknet architecture block list obtained from cfg file
        net_info (dict): contains the network information given in cfg file
        module_list (torch.nn.ModuleList): list of the architecture blocks
        header (torch.IntTensor): contains the header info given in cfg file
        seen (int): images seen by the network

    --------
    Methods:
        get_blocks() -> list:
            returns the Darknet architecture as a list of dictionary object

        get_module_list() -> torch.nn.ModuleList:
            returns the Darknet architecture as a torch.nn.ModuleList object

        forward(x) -> detections (list):
            forward pass method of the Darknet architecture for input tensor x

        train_mode() -> None:
            train mode contextmanager function for the darknet

        load_weights(weight_file_path):
            load pre-trained weights of the Darknet class

        parse_cfg(cfg_file_path) -> blocks (list):
            read the cfg file of the Darknet and returns the block list of
            the architecture of the Darknet network

        create_modules(blocks): -> net_info (dict),
            module_list (torch.nn.ModuleList):
                gets the architecture block list and converts it to the
                torch.nn.ModuleList object to build network
    """

    def __init__(self, cfg_file_path, CUDA):
        """Constructor of the Darknet Class

        --------
        Arguments:
            cfg_file_path (string): file path of the Darknet configure file
        """
        super(Darknet, self).__init__()
        self.blocks = self.parse_cfg(cfg_file_path)
        self.net_info, self.module_list = self.create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.CUDA = CUDA
        self.TRAIN = False

    def get_blocks(self) -> list:
        """Returns the Darknet architecture as a list of dictionary object"""
        return self.blocks

    def get_module_list(self) -> torch.nn.ModuleList:
        """Returns the Darknet architecture as a torch.nn.ModuleList object"""
        return self.module_list

    def forward(self, x) -> list:
        """Forward pass method of the Darknet Class

        --------
        Arguments:
            x (torch.Tensor): input tensor of the Darknet Class
        """
        detections, modules, outputs, write = self.initialize_forward()

        # creating the modules with respect to the block information
        for i in range(len(modules)):
            module_type = (modules[i]["type"])

            # convolutional, upsample or maxpool layers
            if module_type == "convolutional" or module_type \
                    == "upsample" or module_type == "maxpool":
                x = self.pass_through_convolution(i, outputs, x)

            # route layer
            elif module_type == "route":
                x = self.pass_through_route(i, modules, outputs, x)

            # shortcut layer
            elif module_type == "shortcut":
                x = self.pass_through_shortcut(i, modules, outputs, x)

            # detection layer
            elif module_type == 'yolo':
                anchors, inp_dim = self.pass_through_detection(i, modules)

                # output the result
                global CUDA
                x = predict_transform(x, inp_dim,
                                      anchors,
                                      self.num_classes,
                                      self.CUDA,
                                      TRAIN=self.TRAIN)
                if type(x.data) == int:
                    continue
                if not write:
                    self.anchors = anchors.copy()
                    detections = x
                    write = 1
                else:
                    self.anchors.extend(anchors)
                    detections = torch.cat((detections, x), 1)

                del anchors
                outputs[i] = outputs[i-1]

        # if there is no detection it returns 0
        try:
            return detections
        except NameError:
            return 0

    def pass_through_detection(self, i, modules):
        anchors = self.module_list[i][0].anchors
        # get the input dimensions
        inp_dim = int(self.net_info["height"])
        # get the number of classes
        self.num_classes = int(modules[i]["classes"])
        return anchors, inp_dim

    @staticmethod
    def pass_through_shortcut(i, modules, outputs, x):
        from_ = int(modules[i]["from"])
        x = outputs[i - 1] + outputs[i + from_]
        outputs[i] = x
        return x

    @staticmethod
    def pass_through_route(i, modules, outputs, x):
        layers = modules[i]["layers"]
        layers = [int(a) for a in layers]
        if (layers[0]) > 0:
            layers[0] = layers[0] - i
        # layers argument has one value, no concatenation
        if len(layers) == 1:
            x = outputs[i + (layers[0])]

        # layers argument has two values, concatenation
        else:
            if (layers[1]) > 0:
                layers[1] = layers[1] - i

            map1 = outputs[i + layers[0]]
            map2 = outputs[i + layers[1]]

            x = torch.cat((map1, map2), 1)
        outputs[i] = x
        return x

    def pass_through_convolution(self, i, outputs, x):
        x = self.module_list[i](x)
        outputs[i] = x
        return x

    def initialize_forward(self):
        detections = []
        modules = self.blocks[1:]
        # we cache the outputs for the route layer
        outputs = {}
        write = 0
        return detections, modules, outputs, write

    @contextmanager
    def train_mode(self):
        """
            Activates the training mode for the darknet by using with phrase
        """
        try:
            self.TRAIN = True
            yield
        finally:
            self.TRAIN = False

    def load_weights(self, weight_file_path: str):
        """Loads the pre-trained weights for the obtained Darknet Network

        --------
        Arguments:
            weight_file_path (str): path of the binary weight file
        """
        weights = self.configure_weights(weight_file_path)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except KeyError:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    ptr = self.configure_conv_weights_with_batch_norm(model, ptr, weights)
                else:
                    ptr = self.configure_conv_weights_without_batch_norm(conv, ptr, weights)

                # let us load the weights for the Convolutional layers
                ptr = self.configure_conv_weights(conv, ptr, weights)

    @staticmethod
    def configure_conv_weights(conv, ptr, weights):
        num_weights = conv.weight.numel()
        # do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights
        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)
        return ptr

    @staticmethod
    def configure_conv_weights_without_batch_norm(conv, ptr, weights):
        # number of biases
        num_biases = conv.bias.numel()
        # load the weights
        conv_biases = torch.from_numpy(
            weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        # reshape the loaded weights according to the dims of
        # the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)
        # finally copy the data
        conv.bias.data.copy_(conv_biases)
        return ptr

    @staticmethod
    def configure_conv_weights_with_batch_norm(model, ptr, weights):
        bn = model[1]
        # get the number of weights of Batch Norm Layer
        num_bn_biases = bn.bias.numel()
        # load the weights
        bn_biases = torch.from_numpy(
            weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases
        bn_weights = torch.from_numpy(
            weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases
        bn_running_mean = torch.from_numpy(
            weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases
        bn_running_var = torch.from_numpy(
            weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases
        # cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)
        # copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)
        return ptr

    def configure_weights(self, weight_file_path):
        # open the weights file
        fp = open(weight_file_path, "rb")
        # the first 4 values are header information
        # 1. major version number
        # 2. minor Version Number
        # 3. subversion number
        # 4. iMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        return weights

    @staticmethod
    def parse_cfg(cfg_file_path):
        """Configure the Darknet architecture with respect to the given cfg
        file path.

        --------
        Arguments:
            cfg_file_path (str): path of the cfg file
        """
        file = open(cfg_file_path, 'r')
        lines = Darknet.configure_cfg_file(file)
        block = {}
        blocks = []
        Darknet.parse_cfg_line_by_line(block, blocks, lines)
        return blocks

    @staticmethod
    def configure_cfg_file(file: TextIOWrapper):
        lines = file.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#']
        lines = [x.rstrip().lstrip() for x in lines]
        return lines

    @staticmethod
    def parse_cfg_line_by_line(block, blocks, lines):
        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)  # add it the blocks list
                    block = {}  # re-init the block
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

    @staticmethod
    def create_modules(blocks):
        """Builds the architecture of the Darknet network

        --------
        Arguments:
            blocks (list): list of the dictionaries of the blocks
        """
        # captures the information about the input and pre-processing
        index, module_list, net_info, output_filters, prev_filters = Darknet.initialize_blocks(blocks)

        for block in blocks:
            module = nn.Sequential()

            if (block["type"] == "net"):
                continue

            # if it's a convolutional layer
            if (block["type"] == "convolutional"):
                # get the info about the layer
                activation = block["activation"]
                try:
                    batch_normalize = int(block["batch_normalize"])
                    bias = False
                except (ValueError, KeyError):
                    batch_normalize = 0
                    bias = True

                filters = int(block["filters"])
                padding = int(block["pad"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                # add the convolutional layer
                conv = nn.Conv2d(prev_filters, filters,
                                 kernel_size, stride, pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)

                # add the Batch Norm Layer
                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)

                # check the activation.
                # it is either Linear or a Leaky ReLU for YOLO
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module("leaky_{0}".format(index), activn)

            # if it's an upsampling layer
            # we use Bilinear2dUpsampling

            elif (block["type"] == "upsample"):
                Darknet.create_upsample_block(block, index, module)

            # if it is a route layer
            elif (block["type"] == "route"):
                filters = Darknet.create_route_block(block, filters, index, module, output_filters)

            # shortcut corresponds to skip connection
            elif block["type"] == "shortcut":
                Darknet.create_shortcut_block(index, module)

            elif block["type"] == "maxpool":
                Darknet.create_maxpool_block(block, index, module)

            # YOLO is the detection layer
            elif block["type"] == "yolo":
                Darknet.create_yolo_block(block, index, module)

            else:
                print("Unknown block error: A unknown block is provided")
                assert False

            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1

        return net_info, module_list

    @staticmethod
    def create_yolo_block(block, index, module):
        mask = block["mask"].split(",")
        mask = [int(x) for x in mask]
        anchors = block["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i + 1])
                   for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]
        detection = DetectionLayer(anchors)
        module.add_module("Detection_{}".format(index), detection)

    @staticmethod
    def create_maxpool_block(block, index, module):
        stride = int(block["stride"])
        size = int(block["size"])
        if stride != 1:
            maxpool = nn.MaxPool2d(size, stride)
        else:
            maxpool = MaxPoolStride1(size)
        module.add_module("maxpool_{}".format(index), maxpool)

    @staticmethod
    def create_shortcut_block(index, module):
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(index), shortcut)

    @staticmethod
    def create_route_block(block, filters, index, module, output_filters):
        block["layers"] = block["layers"].split(',')
        # start  of a route
        start = int(block["layers"][0])
        # end, if there exists one.
        try:
            end = int(block["layers"][1])
        except IndexError:
            end = 0
        # positive anotation
        if start > 0:
            start = start - index
        if end > 0:
            end = end - index
        route = EmptyLayer()
        module.add_module("route_{0}".format(index), route)
        if end < 0:
            filters = output_filters[index +
                                     start] + \
                      output_filters[index + end]
        else:
            filters = output_filters[index + start]
        return filters

    @staticmethod
    def create_upsample_block(block, index, module):
        stride = int(block["stride"])
        # upsample = Upsample(stride)
        upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        module.add_module("upsample_{}".format(index), upsample)

    @staticmethod
    def initialize_blocks(blocks):
        net_info = blocks[0]
        module_list = nn.ModuleList()
        # indexing blocks helps with implementing route layers
        index = 0
        prev_filters = 3
        output_filters = []
        return index, module_list, net_info, output_filters, prev_filters
