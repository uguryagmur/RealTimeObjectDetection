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


def box_write(tensor, results) -> np.ndarray:
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
    color = random.choice(colors)
    label = "{0} {1:.4}".format(classes[cls], tensor[-2])
    cv2.rectangle(img, c_1, c_2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c_2 = c_1[0] + t_size[0] + 3, c_1[1] + t_size[1] + 4
    cv2.rectangle(img, c_1, c_2, color, -1)
    cv2.putText(img, label, (c_1[0], c_1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, [225, 255, 255], 1)
    return img


def read_directory(directory) -> list:
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


def batch_img_load(img_list, batch_size=1):
    list_end = False
    index = 0
    while True:
        if index + batch_size > len(img_list):
            list_end = True
        if not list_end:
            batch_list = img_list[index: index+batch_size]
        else:
            batch_list = img_list[index:]
        loaded_ims = [cv2.imread(x) for x in batch_list]
        index += batch_size

        im_batches = list(map(prep_image, loaded_ims,
                              [inp_dim for x in range(len(img_list))]))
        im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        if list_end:
            return batch_list, im_batches, im_dim_list, loaded_ims
        else:
            yield batch_list, im_batches, im_dim_list, loaded_ims


if __name__ == '__main__':
    # configuration of argument metrics
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = args.CUDA
    TORCH = args.TORCH
    assert type(CUDA) == bool
    metrics = {}

    # configuration of classes
    num_classes = 80
    classes = load_classes("data/coco.names")

    # configuration of darknet
    print("Loading network.....")
    model = Darknet(args.cfg_file, CUDA)
    if TORCH:
        model.load_state_dict(torch.load(args.weights_file))
    else:
        model.load_weights(args.weights_file)
    print("Network successfully loaded")

    # if the destination path doesn't exist
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    # input dimension check
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # parallel computing adjustment
    if CUDA and torch.cuda.device_count() > 1:
        model == nn.DataParallel(model)
    # if there is only one GPU, use only this one
    elif CUDA:
        model.cuda()

    img_path_list, img_name_list = read_directory(images)
    print('Number of Images= ', len(img_path_list))
    batch_gen = batch_img_load(img_path_list, batch_size)

    for i, batch_data in enumerate(batch_gen):
        print('BATCH NUMBER: ', i)
        batch_list = batch_data[0]
        im_batches = batch_data[1][0]
        im_dim_list = batch_data[2]
        loaded_ims = batch_data[3]
        write = 0
        prediction = torch.zeros(1)
        if CUDA:
            im_batches = im_batches.cuda()
            im_dim_list = im_dim_list.cuda()
        start = time.time()
        with torch.no_grad():
            prediction = model(Variable(im_batches))

        prediction = write_results(prediction,
                                   num_classes,
                                   confidence=confidence,
                                   nms_conf=nms_thesh)

        print(prediction)
        print(im_dim_list)
        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(batch_list):
                # im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------o----------------")

            metrics[img_name_list[i]] = prediction
            continue

        # transform the atribute from index in batch to index in imlist
        prediction[:, 0] += i*batch_size
        output = None

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(batch_list):
            # im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------o----------------")

            # writing the metrics for each file and each class
            text = ""
            for x in output:
                bbox_text = "\t\tBounding Box: {}\n".format(
                    (torch.round(x)[1:5].tolist()))
                text += ''.join(["\tObject:\n"
                                 "\t\tClass: {}\n".format(classes[int(x[-1])]),
                                 bbox_text,
                                 "\t\tObjectness: {:.4f}\n".format(x[-3]),
                                 "\t\tClass Conf.: {:.4f}\n".format(x[-2])])
            metrics[img_name_list[i]] = prediction.tolist()

        if CUDA:
            torch.cuda.synchronize()

        if output is None:
            print("No detections were made")
            exit()

        output[:, 0] = output[:, 0] - i
        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(416/im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor
                              * im_dim_list[:, 0].view(-1, 1))/2
        output[:, [2, 4]] -= (inp_dim - scaling_factor
                              * im_dim_list[:, 1].view(-1, 1))/2

        output[:, 1:5] /= scaling_factor

        for j in range(output.shape[0]):
            output[j, [1, 3]] = torch.clamp(output[j, [1, 3]],
                                            0.0, im_dim_list[j, 0])
            output[j, [2, 4]] = torch.clamp(output[j, [2, 4]],
                                            0.0, im_dim_list[j, 1])

        output_recast = time.time()
        class_load = time.time()
        colors = pkl.load(open("weights/pallete", "rb"))

        draw = time.time()

        list(map(lambda x: box_write(x, loaded_ims), output))

        det_names = pd.Series(img_path_list[i]).apply(
            lambda x: "{}/det_{}_{}".format(args.det,
                                            args.cfg_file[4:-4],
                                            x.split("/")[-1]))

        list(map(cv2.imwrite, det_names, loaded_ims))

    end = time.time()

    # writing the metrics to file
    metrics_file = args.det + '/metrics.yaml'
    with open('metrics.json', 'w') as file:
        json.dump(metrics, file)

    # empty cuda cash
    torch.cuda.empty_cache()
