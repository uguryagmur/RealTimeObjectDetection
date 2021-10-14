'''VOC and COC Dataloader for Object Detection'''

import glob
import json
import torch
import numpy as np
from sys import stderr
from PIL import Image
from PIL import ImageDraw
from contextlib import contextmanager
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset, DataLoader
try:
    from .util import prep_image, xyxy2xywh, draw_boxes
except ImportError:
    from util import prep_image, xyxy2xywh, draw_boxes


class VOC(Dataset):
    r"""Dataset Class for PASCAL VOC Dataset which is used for darknet training

    Attributes:
        xml_directory (str): Directory of the ground truth folder
        img_directory (str): Directory of the images in the dataset
        resolution (int): Input image dimensions of the darknet
        fformat (str): format of the image files (default='.jpg')
        train_set (torch.utils.data.Subset): Training set
        val_set (torch.utils.data.Subset): Validation set
        train_set_loader (DataLoader): Training set loader
        val_set_loader (DataLoader): Validation set loader
        split (Bool): whether the dataset is splitted to train and valid sets
    """

    def __init__(self, xml_directory, img_directory, resolution=416,
                 fformat='.jpg') -> None:
        r"""
        Constructor of the VOCDataset Class
        """

        assert isinstance(fformat, str)
        assert isinstance(resolution, (int))
        self.xml_path_list = glob.glob(xml_directory+'/*'+'.xml')
        self.resolution = resolution
        self.split = False
        if self.xml_path_list == []:
            raise FileNotFoundError("""FileNotFoundError: For the given
directory {} and file format {}, no file was
found.""".format(xml_directory, fformat))
        self.data = dict()
        for element in self.xml_path_list:
            value = img_directory + '/' + element[-15:-4] + fformat
            self.data[element] = value
        print('{} number of given data is loaded and ready!\n'
              .format(len(self.xml_path_list)))

    def __len__(self) -> int:
        r"""The function to learn length of the adjusted dataset

        Returns:
            Integer: Length of the dataset
        """

        return len(self.data)

    def read_xml(self, filename) -> dict:
        r"""The function to read xml file and extract ground truth
        information for PASCAL VOC Dataset

        Parameters:
            filename (str): destination of the xml file

        Returns:
            List: Bounding box of objects (person)
        """

        doc = ET.parse(filename).getroot()
        bboxes = []

        self.fetch_boxes_from_xml(bboxes, doc)

        if bboxes == []:
            return None
        else:
            return bboxes

    @staticmethod
    def fetch_boxes_from_xml(bboxes, doc):
        for elem in doc.findall('object'):

            # because we want only person detections
            if elem.find('name').text == 'person':
                bboxes.append([float(elem.find('bndbox/xmin').text),
                               float(elem.find('bndbox/ymin').text),
                               float(elem.find('bndbox/xmax').text),
                               float(elem.find('bndbox/ymax').text)])

    def __getitem__(self, i):
        r"""The function to get an item from the dataset

        Parameters:
            i (int): index integer to get file from list

        Returns:
            torch.tensor: Given image data in a torch.tensor form
        """

        assert isinstance(i, int)
        assert i < len(self.xml_path_list)
        bbox, img = self.load_image(i)
        pad, ratio = self.configure_image(img)
        if bbox is not None:
            bbox = self.configure_boun_box(bbox, pad, ratio)
        img = np.asarray(img)
        img = prep_image(img, self.resolution, mode='RGB').squeeze(0)
        return img, bbox

    def configure_image(self, img):
        max_im_size = max(img.size)
        w, h = img.size
        ratio = float(self.resolution / max_im_size)
        pad = [int((max_im_size - w) * ratio / 2), int((max_im_size - h) * ratio / 2)]
        return pad, ratio

    def load_image(self, i):
        bbox = self.read_xml(self.xml_path_list[i])
        img_path = self.data[self.xml_path_list[i]]
        img = Image.open(img_path)
        return bbox, img

    @staticmethod
    def configure_boun_box(bbox, pad, ratio):
        for b in bbox:
            b.extend([1, 1])
            b.extend([0] * 79)
        bbox = torch.tensor(bbox)
        bbox = xyxy2xywh(bbox)
        bbox[..., :4] *= ratio
        bbox[:, 0] += pad[0]
        bbox[:, 1] += pad[1]
        return bbox

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for the dataloader of the dataset

        Parameters:
            batch (list): data samples of the current batch

        Returns:
            img (torch.Tensor): image samples of the current batch
            bndbox (list): list of bounding box tensors for every image
        """
        img, bndbox = zip(*batch)
        img = torch.stack(img, dim=0)
        return img, bndbox

    def get_dataloader(self, batch_size, shuffle=True,
                       num_workers=4) -> DataLoader:
        r"""The function to create a dataloader for the dataset class

        Parameters:
            batch_size (int): Batch size of the training set
            shuffle (bool): Whether you want shuffling or not
            split (bool): If the dataset is splitted, it returns 2 dataloaders
            num_workers (int): Number of subprocesses to use for data loading.

        Returns:
            DataLoader, DataLoader: torch DataLoader object for training and
            validation sets
        """

        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=self.collate_fn,
                          num_workers=num_workers)


class COCO(Dataset):
    """COCO Dataset DataLoader for Object Detection

    Attributes:
        img_ids (list): list of image ids of the COCO dataset
        img_annotations (dict): dictionary of the image annotations
        images (dict): information about images and their URLs
        resolution (int): resolution of the training
        img_dir (str): path of the folder containing the COCO images
        deleted_cls (list): list of the deletec class for the corresponding dataset
        keep_img_name (bool): flag to return image names for each sample
        only_gt (bool): flag to return ground truth of the images without image data
    """

    def __init__(self, anotations_json, img_dir,
                 resolution=416, keep_img_name=False,
                 only_ground_truth=False):
        '''Constructor of COCO Class'''

        super(COCO, self).__init__()
        self.resolution = resolution
        self.img_dir = img_dir
        if self.img_dir[-1] != '/':
            self.img_dir += '/'
        self.read_annotations(anotations_json)
        self.deleted_cls = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
        self.keep_img_name = keep_img_name
        self.only_gt = only_ground_truth

    def read_annotations(self, anotations_json, non_crowd=True):
        """The method to read annotation files of the COCO dataset
        and store them in the dictionary and list objects

        Parameters:
            anotations_json (str): annotation file directory
            non_crowd (bool): flag to choose only non_crowd images
        """

        ann = json.load(open(anotations_json))
        if non_crowd:
            img_ids = [i['image_id'] for i in ann['annotations']
                       if not i['iscrowd']]
        else:
            img_ids = [i['image_id'] for i in ann['annotations']]
        self.img_ids = list(set(img_ids))
        self.img_annotations = ann['annotations']
        self.images = {i['id']: i for i in ann['images']}

    def coco2yolo(self, category_id):
        """This function converts the COCO dataset labels for the corresponding
        Darknet YOLO detector network label

        Parameters:
            category_id (int): category_id label for the corresponding bounding box
        """
        ex = 0
        for i in range(len(self.deleted_cls)):
            if category_id < self.deleted_cls[i]:
                return category_id - ex
            ex += 1
        if category_id - ex < 0:
            print('CATEGORY_ID ERROR', file=stderr)
            exit()
        return category_id - ex

    def __len__(self):
        r"""The function to learn length of the adjusted dataset

        Returns:
            Integer: Length of the dataset
        """
        return len(self.img_ids)

    def __getitem__(self, index):
        r"""The function to get an item from the dataset

        Parameters:
            i (int): index integer to get file from list

        Returns:
            torch.tensor: Given image data in a torch.tensor form
        """

        id_, img = self.fetch_image(index)
        if self.keep_img_name:
            img_name = self.images[id_]['file_name']

        pad, ratio = self.configure_padding(img)
        if not self.only_gt:
            img = np.asarray(img)
            img = prep_image(img, self.resolution, mode='RGB').squeeze(0)

        bbox = self.fetch_bounding_boxes(id_, pad, ratio)

        # draw_boxes(img, bbox, 'coco_val_with_box/'+img_name)
        if bbox != []:
            bbox = torch.stack(bbox, dim=0)
        if not self.keep_img_name:
            if not self.only_gt:
                return img, bbox
            else:
                return bbox

        else:
            if not self.only_gt:
                return img_name, img, bbox
            else:
                return img_name, bbox

    def fetch_bounding_boxes(self, id_, pad, ratio):
        bbox = []
        for annot in self.img_annotations:
            if annot['image_id'] == id_:
                cls_encoding = [1.0]
                cls_encoding.extend([0] * 80)
                # print(obj['category_id'], self.coco2yolo(obj['category_id']))
                cls_encoding[self.coco2yolo(annot['category_id'])] = 1.0
                box = annot['bbox'][:5]
                box.extend(cls_encoding)
                box = torch.FloatTensor(box)
                box[:4] *= ratio
                box[0] += box[2] / 2 + pad[0]
                box[1] += box[3] / 2 + pad[1]
                bbox.append(box)
        return bbox

    def configure_padding(self, img):
        # obtaining the image size
        w, h = img.size
        max_im_size = max(w, h)
        ratio = float(self.resolution / max_im_size)
        # calculating paddings for bboxes
        pad = [int((max_im_size - w) * ratio / 2), int((max_im_size - h) * ratio / 2)]
        return pad, ratio

    def fetch_image(self, index):
        id_ = self.img_ids[index]
        img = self.img_dir + self.images[id_]['file_name']
        img = Image.open(img).convert('RGB')
        return id_, img

    def collate_fn(self, batch):
        """
        Collate function for the dataloader of the dataset

        Parameters:
            batch (list): data samples of the current batch

        Returns:
            img (torch.Tensor): image samples of the current batch
            bndbox (list): list of bounding box tensors for every image
        """
        if not self.only_gt:
            if not self.keep_img_name:
                img, bbox = zip(*batch)
                img = torch.stack(img, dim=0)
                return img, bbox
            else:
                img_name, img, bbox = zip(*batch)
                img = torch.stack(img, dim=0)
                return img_name, img, bbox
        else:
            if not self.keep_img_name:
                bbox = zip(*batch)
                return bbox
            else:
                img_name, bbox = zip(*batch)
                return img_name, bbox

    @contextmanager
    def only_ground_truth(self):
        """Activates the only ground truth mode for the COCO dataset in which
        dataloader only load the ground truth of the corresponding images
        """
        try:
            self.only_gt = True
            yield
        finally:
            self.only_gt = False

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        r"""The function to create a dataloader for the dataset class

        Parameters:
            batch_size (int): Batch size of the training set
            shuffle (bool): Whether you want shuffling or not
            split (bool): If the dataset is splitted, it returns 2 dataloaders
            num_workers (int): Number of subprocesses to use for data loading.

        Returns:
            DataLoader, DataLoader: torch DataLoader object for training and
            validation sets
        """
        dloader = DataLoader(self, batch_size=batch_size,
                             collate_fn=self.collate_fn,
                             shuffle=shuffle,
                             num_workers=num_workers)
        return dloader


if __name__ == '__main__':
    # dataset testing codes
    # for VOC dataset
    xml_dir = '/home/adm1n/Datasets/SPAutoencoder/\
VOCdevkit/VOC2012/Annotations'
    img_dir = '/home/adm1n/Datasets/SPAutoencoder/VOC2012'

    # for COCO dataset
    json_path = '/home/adm1n/Datasets/\
COCO/2017/annotations/instances_val2017.json'
    img_path = '/home/adm1n/Datasets/COCO/2017/val2017/'

    Dset = VOC(xml_dir, img_dir)
    dset = COCO(json_path, img_path)
    # print(dset.__len__())
    # print(Dset.__len__())
    img, bbox = dset.__getitem__(7)
    img = img.transpose(0, 1).transpose(1, 2).numpy()
    img = Image.fromarray(np.uint8(img*255))
    draw = ImageDraw.Draw(img)
    for b in bbox:
        if b[5] != 1:
            pass
        box = b[:4].numpy()
        bbox = [0, 0, 0, 0]
        bbox[0] = int(box[0] - box[2]/2)
        bbox[1] = int(box[1] - box[3]/2)
        bbox[2] = int(box[0] + box[2]/2)
        bbox[3] = int(box[1] + box[3]/2)
        draw.rectangle(bbox, outline='red')
    img.show()
