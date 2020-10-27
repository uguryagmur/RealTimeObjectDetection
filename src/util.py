"""DOCSTRING will be added later"""
from __future__ import division

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw


def xyxy2xywh(box: torch.Tensor) -> torch.Tensor:
    """DOCSTRING will be added later"""
    output = torch.zeros(box.size())
    output[..., 0] = (box[..., 2] + box[..., 0])/2
    output[..., 1] = (box[..., 3] + box[..., 1])/2
    output[..., 2] = box[..., 2] - box[..., 0]
    output[..., 3] = box[..., 3] - box[..., 1]
    output[..., 4:] = box[..., 4:]
    return output


def xywh2xyxy(box: torch.Tensor) -> torch.Tensor:
    """DOCSTRING will be added later"""
    output = torch.zeros(box.size())
    output[..., 0] = (box[..., 0] - box[..., 2]/2)
    output[..., 1] = (box[..., 1] - box[..., 3]/2)
    output[..., 2] = (box[..., 0] + box[..., 2]/2)
    output[..., 3] = (box[..., 1] + box[..., 3]/2)
    output[..., 4:] = box[..., 4:]
    return output


def xywh2YOLO(box: torch.Tensor, stride: float,
              anchor: tuple):
    """DOCSTRING will be added later"""
    x = box[..., 0].item()/stride
    x_coor = int(box[..., 0].item()/stride)
    y = box[..., 1].item()/stride
    y_coor = int(box[..., 1].item()/stride)
    x -= x_coor
    y -= y_coor
    w = torch.log(box[..., 2] / anchor[0] + 1e-16).item()
    h = torch.log(box[..., 3] / anchor[1] + 1e-16).item()
    return y_coor, x_coor, y, x, w, h


def draw_boxes(img: torch.Tensor, bbox: torch.Tensor,
               img_dir, from_tensor=False):
    if from_tensor:
        img = img.transpose(0, 1).transpose(1, 2).numpy()*255
        img = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(img)
    for b in bbox:
        if b[5] != 1:
            continue
        box = b[:4].numpy()
        bbox = [0, 0, 0, 0]
        bbox[0] = int(box[0] - box[2]/2)
        bbox[1] = int(box[1] - box[3]/2)
        bbox[2] = int(box[0] + box[2]/2)
        bbox[3] = int(box[1] + box[3]/2)
        draw.rectangle(bbox, outline='red')
    img.show()
    # img.save(img_dir)


def confidence_mask(tensor: torch.Tensor, confidence: float) -> torch.Tensor:
    """DOCSTRING will be added later"""
    # confidence masking for direct output of the YOLO layer
    conf_mask = (tensor[:, :, 4] > confidence).float().unsqueeze(2)
    return tensor*conf_mask


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Returns the IoU of two bounding boxes

    Arguments:
        box1 (torch.Tensor) : coor tensor of the first box to calculate IoU
        box2 (torch.Tensor) : coor tensor of the first box to calculate IoU
    """
    # get the coordinates of bounding boxes
    b1_x1, b1_y1 = box1[..., 0], box1[..., 1]
    b1_x2, b1_y2 = box1[..., 2], box1[..., 3]
    b2_x1, b2_y1 = box2[..., 0], box2[..., 1]
    b2_x2, b2_y2 = box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def bbox_iou_wh(wh1: tuple, wh2: tuple) -> float:
    """DOCSTRING will be added later"""
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[0]
    intersect_area = min(w1, w2) * min(h1, h2)
    union_area = w1*h1 + w2*h2 - intersect_area
    return intersect_area/union_area


# this function is fixed and won't cut the autograd backward graph
def predict_transform(prediction, inp_dim, anchors, num_class,
                      CUDA, TRAIN=False) -> torch.Tensor:
    """
    Returns the prediction tensor with respect to the output of the YOLO
    Detection Layer

    Arguements:
        prediction (torch.Tensor) : output tensor of the Detection Layer
        inp_dim (torch.Tensor) : input image dimensions
        anchors (torch.Tensor) : anchors of the Darknet
        num_class (int) : number of classes can be detected by Darknet
    """

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_class
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # sigmoid the  centre_x, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4:] = torch.sigmoid(prediction[:, :, 4:])

    # for trianing no offset addition
    if not TRAIN:
        # decreasing the size of the anchor boxes to match with prediction
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        # log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        # add the center offsets
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if CUDA:
            anchors = anchors.cuda()
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
            1, num_anchors).view(-1, 2).unsqueeze(0)

        prediction[:, :, :2] += x_y_offset

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors
        prediction[:, :, :4] *= stride

    return prediction  # -> shape = [batch, #_of_boxes, 5 + #_of_classes]


def write_results(prediction, num_class, confidence=0.6,
                  nms_conf=0.4) -> torch.Tensor:
    """
    Returns the results of the predictions of the Darknet
    as bounding boxes and class of the object

    Arguments:
        prediction (torch.Tensor) : prediction output of the
                    predict_transform function
        confidence (float) : confidence of the detection
        num_class (int) : number of classes can be detected by Darknet
        nms_conf (float) : non-max supression confidence (default=0.4)
    """

    # confidence masking
    prediction = confidence_mask(prediction, confidence)

    # transforming box attributes to corner coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # obtaining the batch_size of the input tensor
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        # generating the max confidence sequence
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 +
                                                        num_class], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except IndexError:
            continue

        # if there is no detection for this batch image
        if image_pred_.shape[0] == 0:
            continue

        # get the various classes detected in the image
        # -1 index holds the class index
        img_classes = torch.unique(image_pred_[:, -1])

        for cls in img_classes:
            # perform NMS
            # get the detections with one particular class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sorting the detections
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # get the IOUs of all boxes
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                                    0), image_pred_class[i+1:])
                except (IndexError, ValueError):
                    break

                # zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # repeat the batch for as many detections of the class in the img
            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except NameError:
        return 0


def letterbox_image(img, inp_dim) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding

    Arguments:
        img (np.ndarray) : image which will be reshaped as numpy array
        inp_dim (list) : Darknet input image dimensions
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
           2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim, mode='BGR') -> torch.Tensor:
    """
    Prepare image for inputting to the neural network as torch.Tensor

    Arguments:
        img (np.ndarray) : input image as numpy array form
        inp_dim (list) : input image dimensions
    """

    assert mode == 'BGR' or mode == 'RGB'
    # scaling image by protecting the aspect-ratio
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    # transpose for the pytorch tensor
    if mode == 'RGB':
        img = img.transpose((2, 0, 1)).copy()
    else:
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # normalization of the image
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(names_file_path: str) -> list:
    """Load the classes from dataset file

    Arguments:
        names_file_path (str) : path of the names file of the dataset
    """
    fp = open(names_file_path, "r")
    names = fp.read().split("\n")[:-1]
    return names
