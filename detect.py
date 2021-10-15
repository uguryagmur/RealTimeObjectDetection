from __future__ import division
import os
import cv2
import json
import time
import torch
import random
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
import os.path as osp
import torch.nn as nn
from src.darknet import Darknet
from torch.autograd import Variable
from src.util import load_classes, prep_image, write_results


class DarknetDetector:
    def __init__(self):
        # configuration of argument metrics
        self.args = arg_parse()
        self.parse_arguments(self.args)
        assert type(self.CUDA) == bool
        self.metrics = {}
        self.num_classes = 80
        self.classes = load_classes("data/coco.names")

    def parse_arguments(self, args):
        self.images = args.images
        self.batch_size = int(args.bs)
        self.confidence = float(args.confidence)
        self.nms_thresh = float(args.nms_thresh)
        self.CUDA = args.CUDA
        self.TORCH = args.TORCH

    def __call__(self, *args, **kwargs):
        model = self.configure_darknet()

        # if the destination path doesn't exist
        if not os.path.exists(self.args.det):
            os.makedirs(self.args.det)

        model.net_info["height"] = self.args.reso
        self.inp_dim = int(model.net_info["height"])

        # input dimension check
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        img_path_list, img_name_list = DarknetDetector.read_directory(self.images)
        print('Number of Images= ', len(img_path_list))
        batch_gen = self.batch_img_load(img_path_list, self.batch_size)

        for i, batch_data in enumerate(batch_gen):
            print('BATCH NUMBER: ', i)
            batch_list = batch_data[0]
            im_batches = batch_data[1][0]
            im_dim_list = batch_data[2]
            loaded_ims = batch_data[3]
            write = 0
            prediction = torch.zeros(1)
            if self.CUDA:
                im_batches = im_batches.cuda()
                im_dim_list = im_dim_list.cuda()
            start = time.time()
            with torch.no_grad():
                prediction = model(Variable(im_batches))

            prediction = write_results(prediction,
                                       self.num_classes,
                                       confidence=self.confidence,
                                       nms_conf=self.nms_thresh)

            print(prediction)
            print(im_dim_list)
            end = time.time()

            if type(prediction) == int:

                for im_num, image in enumerate(batch_list):
                    # im_id = i*batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds".format(
                        image.split("/")[-1], (end - start) / self.batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------o----------------")

                self.metrics[img_name_list[i]] = prediction
                continue

            # transform the attribute from index in batch to index in imlist
            prediction[:, 0] += i * self.batch_size
            output = None

            if not write:  # If we haven't initialised output
                output = prediction
                write = 1
            else:
                output = torch.cat((output, prediction))

            for im_num, image in enumerate(batch_list):
                # im_id = i*batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in output]
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start) / self.batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------o----------------")

                # writing the metrics for each file and each class
                text = ""
                for x in output:
                    bbox_text = "\t\tBounding Box: {}\n".format(
                        (torch.round(x)[1:5].tolist()))
                    text += ''.join(["\tObject:\n"
                                     "\t\tClass: {}\n".format(self.classes[int(x[-1])]),
                                     bbox_text,
                                     "\t\tObjectness: {:.4f}\n".format(x[-3]),
                                     "\t\tClass Conf.: {:.4f}\n".format(x[-2])])
                self.metrics[img_name_list[i]] = prediction.tolist()

            if self.CUDA:
                torch.cuda.synchronize()

            if output is None:
                print("No detections were made")
                exit()

            output[:, 0] = output[:, 0] - i
            im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

            scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (self.inp_dim - scaling_factor
                                  * im_dim_list[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (self.inp_dim - scaling_factor
                                  * im_dim_list[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for j in range(output.shape[0]):
                output[j, [1, 3]] = torch.clamp(output[j, [1, 3]],
                                                0.0, im_dim_list[j, 0])
                output[j, [2, 4]] = torch.clamp(output[j, [2, 4]],
                                                0.0, im_dim_list[j, 1])

            output_recast = time.time()
            class_load = time.time()
            self.colors = pkl.load(open("weights/pallete", "rb"))

            draw = time.time()

            list(map(lambda x: self.box_write(x, loaded_ims), output))

            det_names = pd.Series(img_path_list[i]).apply(
                lambda x: "{}/det_{}_{}".format(self.args.det,
                                                self.args.cfg_file[4:-4],
                                                x.split("/")[-1]))

            list(map(cv2.imwrite, det_names, loaded_ims))

        end = time.time()

        # writing the metrics to file
        metrics_file = self.args.det + '/metrics.yaml'
        with open('metrics.json', 'w') as file:
            json.dump(self.metrics, file)

        # empty cuda cash
        torch.cuda.empty_cache()

    def use_model_parallelism(self, model):
        # parallel computing adjustment
        if self.CUDA and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        elif self.CUDA:
            model.cuda()
        return model

    def configure_darknet(self):
        print("Loading network.....")
        model = Darknet(self.args.cfg_file, self.CUDA)
        if self.TORCH:
            model.load_state_dict(torch.load(self.args.weights_file))
        else:
            model.load_weights(self.args.weights_file)
        print("Network successfully loaded")
        model = self.use_model_parallelism(model)
        return model

    def box_write(self, tensor, results) -> np.ndarray:
        """Returns the image where the object detection bounding boxes, labels
        and class confidences are printed on

        Arguments:
            tensor (torch.Tensor) : output results of the Darknet detection
            results (torch.Tensor) : loaded images tensor with a certain batch
        """
        c_1 = (tensor[1].int().item(), tensor[2].int().item())
        c_2 = (tensor[3].int().item(), tensor[4].int().item())
        img = results[int(tensor[0])]
        cls = int(tensor[-1])
        color = random.choice(self.colors)
        label = "{0} {1:.4}".format(self.classes[cls], tensor[-2])
        cv2.rectangle(img, c_1, c_2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c_2 = c_1[0] + t_size[0] + 3, c_1[1] + t_size[1] + 4
        cv2.rectangle(img, c_1, c_2, color, -1)
        cv2.putText(img, label, (c_1[0], c_1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, [225, 255, 255], 1)
        return img

    @staticmethod
    def read_directory(directory):
        try:
            img_path_list = [osp.join(osp.realpath('.'), directory, img)
                             for img in os.listdir(directory)]
            img_name_list = [img for img in os.listdir(directory)]
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), directory))
        except FileNotFoundError:
            print("No file or directory with the name {}".format(directory))
            raise
        return img_path_list, img_name_list

    def batch_img_load(self, img_list, batch_size=1):
        list_end = False
        index = 0
        while True:
            if index + batch_size > len(img_list):
                list_end = True
            if not list_end:
                batch_list = img_list[index: index + batch_size]
            else:
                batch_list = img_list[index:]
            loaded_ims = [cv2.imread(x) for x in batch_list]
            index += batch_size

            im_batches = list(map(prep_image, loaded_ims,
                                  [self.inp_dim for x in range(len(img_list))]))
            im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
            if list_end:
                return batch_list, im_batches, im_dim_list, loaded_ims
            else:
                yield batch_list, im_batches, im_dim_list, loaded_ims


def arg_parse():
    """Detect file argument configuration"""

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images",
                        help="""Image / Directory containing
                                images to perform detection upon""",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det',
                        help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size",
                        default=1, type=int)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.6, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.5)
    parser.add_argument("--cfg", dest='cfg_file', help="Config file",
                        default="cfg/yolov3-tiny.cfg", type=str)
    parser.add_argument("--weights", dest='weights_file', help="weightsfile",
                        default="weights/yolov3-tiny.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="""Input resolution of the network. Increase to
                        increase accuracy. Decrease to increase speed""",
                        default=416, type=int)
    parser.add_argument("--use_GPU", dest='CUDA', action='store_true',
                        help="GPU Acceleration Enable Flag (true/false)")
    parser.add_argument("--trained", dest='TORCH', action='store_true',
                        help="Whether torch trained weights file (true/false)")

    return parser.parse_args()


if __name__ == '__main__':
    detector = DarknetDetector()
    detector()
