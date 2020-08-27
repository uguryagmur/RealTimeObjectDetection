""" Dataset and Dataloader for the SuperPixel Autoencoder """

import torch
import glob
import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
try:
    from .util import prep_image
except ImportError:
    from util import prep_image


class VOCDataset(Dataset):
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
                bboxes.append([int(elem.find('bndbox/xmin').text),
                               int(elem.find('bndbox/ymin').text),
                               int(elem.find('bndbox/xmax').text),
                               int(elem.find('bndbox/ymax').text)])

        if bboxes == []:
            return None
        else:
            return bboxes

    def sort_bboxes(self, bndbox: torch.Tensor,
                    descending=True) -> torch.Tensor:
        if bndbox.shape[0] == 1:
            return bndbox
        _, ind = bndbox.sort(dim=0, descending=descending)
        output = bndbox.new(bndbox.shape)
        for i in range(bndbox.shape[0]):
            output[i] = bndbox[ind[i, 0]]
        return output

    def __getitem__(self, i) -> torch.Tensor:
        r"""The function to get an item from the dataset

        Parameters:
            i (int): index integer to get file from list

        Returns:
            torch.tensor: Given image data in a torch.tensor form
        """

        assert isinstance(i, int)
        assert i < len(self.xml_path_list)
        bndbox = self.read_xml(self.xml_path_list[i])
        if bndbox is not None:
            bndbox = torch.tensor(bndbox)
        bndbox = self.sort_bboxes(bndbox)
        img_path = self.data[self.xml_path_list[i]]
        img = Image.open(img_path)
        img = np.asarray(img)
        img = prep_image(img, self.resolution)
        return img[0], bndbox

    @staticmethod
    def collate_fn(batch):
        img, bndbox = zip(*batch)
        img = torch.stack(img, dim=0)
        return img, bndbox

    def random_spilt(self, ratio=0.1) -> None:
        r""" The function to split dataset to train and validation sets

        Splitted dataset will be written on the class attributes

        Parameters:
            ratio (float): ratio of validation to whole dataset
        """

        assert isinstance(ratio, float) and ratio < 1.0
        length = self.__len__()
        offset = int(length*ratio)
        lengths = [offset, length-offset]
        self.train_set, self.val_set = random_split(self, lengths)
        self.split = True
        print('The Dataset is splitted as validation\
and training sets succesfully\n')

    def get_loader(self, batch_size, shuffle=True,
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

        if self.split:
            val_loader = DataLoader(self.train_set, batch_size=batch_size,
                                    collate_fn=self.collate_fn,
                                    shuffle=shuffle, num_workers=num_workers)
            train_loader = DataLoader(self.val_set, batch_size=batch_size,
                                      collate_fn=self.collate_fn,
                                      shuffle=shuffle, num_workers=num_workers)
            return train_loader, val_loader
        else:
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=self.collate_fn,
                              num_workers=num_workers)


if __name__ == '__main__':
    print('SuperPixel Encoding Dataloader Class Trial\n')
    xml_dir = '/home/adm1n/Datasets/SPAutoencoder/\
VOCdevkit/VOC2012/Annotations'
    img_dir = '/home/adm1n/Datasets/SPAutoencoder/VOC2012'
    DSet = VOCDataset(xml_dir, img_dir)
    # img, bndbox = DSet.__getitem__(1)
    # print(bndbox)
    # print('--------o--------')
    # print(img)
    # print('--------o--------')
    VOCLoader = DSet.get_loader(batch_size=3, shuffle=True)
    loader = iter(VOCLoader)
    img, bndbox = loader.next()
    # print('BOUNDING BOXES')
    # print(bndbox)
    # print('--------o--------')
    # print('IMGAES')
    # for i in img:
    #     print(i.size())
    # print(type(img))
    print('-----------------------------------------------------------------')
    for batch, sample in enumerate(VOCLoader):
        print('--------o--------')
        print(sample[1])
    #     print('--------o--------')
    #     print(batch)
        break
