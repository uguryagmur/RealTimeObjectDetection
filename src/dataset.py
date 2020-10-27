'''VOC and COC Dataloader for Object Detection'''

import glob
import json
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
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

        for elem in doc.findall('object'):

            # because we want only person detections
            if elem.find('name').text == 'person':
                bboxes.append([float(elem.find('bndbox/xmin').text),
                               float(elem.find('bndbox/ymin').text),
                               float(elem.find('bndbox/xmax').text),
                               float(elem.find('bndbox/ymax').text)])

        if bboxes == []:
            return None
        else:
            return bboxes

    def __getitem__(self, i) -> torch.Tensor:
        r"""The function to get an item from the dataset

        Parameters:
            i (int): index integer to get file from list

        Returns:
            torch.tensor: Given image data in a torch.tensor form
        """

        assert isinstance(i, int)
        assert i < len(self.xml_path_list)
        bbox = self.read_xml(self.xml_path_list[i])
        img_path = self.data[self.xml_path_list[i]]
        img = Image.open(img_path)
        max_im_size = max(img.size)
        w, h = img.size
        ratio = float(self.resolution/max_im_size)
        pad = [int((max_im_size - w)*ratio/2), int((max_im_size - h)*ratio/2)]
        if bbox is not None:
            for b in bbox:
                b.extend([1, 1])
                b.extend([0]*79)
            bbox = torch.tensor(bbox)
            bbox = xyxy2xywh(bbox)
            bbox[..., :4] *= ratio
            bbox[:, 0] += pad[0]
            bbox[:, 1] += pad[1]
        img = np.asarray(img)
        img = prep_image(img, self.resolution, mode='RGB').squeeze(0)
        return img, bbox

    @staticmethod
    def collate_fn(batch):
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
    '''COCO Dataset DataLoader for Object Detection'''

    def __init__(self, anotations_json, img_dir,
                 resolution=416, keep_img_name=False):
        '''Constructor of COCO Class'''

        super(COCO, self).__init__()
        self.resolution = resolution
        self.img_dir = img_dir
        if self.img_dir[-1] != '/':
            self.img_dir += '/'
        self.read_annotations(anotations_json)
        self.deleted_cls = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
        self.keep_img_name = keep_img_name
        # with open('COCO_val_objs.json', 'w') as file:
        #     json.dump(self.objs, file)
        # with open('COCO_val_img_ids.json', 'w') as file:
        #     json.dump(self.img_ids, file)

    def read_annotations(self, anotations_json, non_crowd=True):
        ann = json.load(open(anotations_json))
        self.img_set = {i['id']: i for i in ann['images']}
        if non_crowd:
            self.objs = [i for i in ann['annotations']
                         if not i['iscrowd']]
            self.img_ids = [i['image_id'] for i in ann['annotations']
                            if not i['iscrowd']]
        else:
            self.objs = [i for i in ann['annotations']]
            self.img_ids = [i['image_id'] for i in ann['annotations']]

    def coco2yolo(self, category_id):
        ex = 0
        for i in range(len(self.deleted_cls)):
            if category_id < self.deleted_cls[i]:
                return category_id - ex
            ex += 1
        if category_id - ex < 0:
            print('CATEGORY_ID ERROR')
            exit()
        return category_id - ex

    def __len__(self):
        return len(self.img_set)

    def __getitem__(self, index):
        id_ = self.img_ids[index]
        img = self.img_dir + self.img_set[id_]['file_name']
        img = Image.open(img).convert('RGB')
        if self.keep_img_name:
            img_name = self.img_set[id_]['file_name']

        # obtaining the image size
        max_im_size = max(img.size)
        w, h = img.size
        ratio = float(self.resolution/max_im_size)

        # calculating paddings for bboxes
        pad = [int((max_im_size - w)*ratio/2), int((max_im_size - h)*ratio/2)]
        img = np.asarray(img)
        img = prep_image(img, self.resolution, mode='RGB').squeeze(0)

        bbox = []
        for obj in self.objs:
            if obj['image_id'] == id_:
                cls_encoding = [1.0]
                cls_encoding.extend([0]*80)
                # print(obj['category_id'], self.coco2yolo(obj['category_id']))
                cls_encoding[self.coco2yolo(obj['category_id'])] = 1.0
                box = obj['bbox'][:5]
                box.extend(cls_encoding)
                box = torch.FloatTensor(box)
                box[:4] *= ratio
                box[0] += box[2]/2 + pad[0]
                box[1] += box[3]/2 + pad[1]
                bbox.append(box)
        # draw_boxes(img, bbox, 'coco_val_with_box/'+img_name)
        if not self.keep_img_name:
            if bbox == []:
                return []
            else:
                bbox = torch.stack(bbox, dim=0)
                return img, bbox

        else:
            if bbox == []:
                return img_name, []
            else:
                bbox = torch.stack(bbox, dim=0)
                return img_name, img, bbox

    def collate_fn(self, batch):
        if not self.keep_img_name:
            img, bbox = zip(*batch)
            img = torch.stack(img, dim=0)
            return img, bbox
        else:
            img_name, img, bbox = zip(*batch)
            img = torch.stack(img, dim=0)
            return img_name, img, bbox

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        dloader = DataLoader(self, batch_size=batch_size,
                             collate_fn=self.collate_fn,
                             shuffle=shuffle,
                             num_workers=num_workers)
        return dloader


if __name__ == '__main__':
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
    img, bbox = dset.__getitem__(6)
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
    # img, bbox = Dset.__getitem__(13)
    # print(img.shape)
    # print('--------o--------')
    # print(bbox)
    # dloader = dset.get_dataloader(batch_size=4)
    # diter = iter(dloader)
    # img, bbox = diter.next()
    # print(img.shape)
    # print('--------o--------')
    # print(len(bbox))
    # for img, bbox in diter:
    # pass
