"""YOLOv3 Darknet Trainer of the Network"""

import time
import torch
import argparse
import torch.nn as nn
from math import ceil
import torch.optim as optim
import matplotlib.pyplot as plt
from src.darknet import Darknet
from src.dataset import COCO
from src.util import predict_transform


class DarknetTrainer():
    """Darknet YOLOv3 Network Trainer Class

    Attributes:
        img_size (list, tuple): Size of the training images
        epoch (int): Epoch number of the training
        batch_size (int): Size of the mini-batches
        dataset (SPDataset): Dataset to train network
        train_loader (DataLoader): torch DataLoader object for training set
        val_loader (DataLoader): torch DataLoader object for validation set
        autoencoder (SPAutoencoder): Autoencoder Network to train
        optimizer (torch.optim): Optimizer to train network
        criterion (torch.nn.MSELoss): Criterion for the loss of network output
        device (torch.device): Device running the training process
    """

    def __init__(self, cfg_file: str, weights_file=None,
                 epoch=10, batch_size=16, resolution=416,
                 confidence=0.6, num_classes=80, CUDA=False) -> None:
        """ Constructor of the Darknet Trainer Class """

        assert isinstance(weights_file, (str, None))
        assert isinstance(epoch, int)
        assert isinstance(batch_size, int)
        assert isinstance(resolution, int)
        assert resolution % 32 == 0
        self.CUDA = bool(torch.cuda.is_available() and CUDA)
        self.num_classes = num_classes
        self.epoch = epoch
        self.batch_size = batch_size
        self.resolution = resolution
        self.confidence = confidence
        self.criterion = nn.functional.mse_loss
        self.darknet = Darknet(cfg_file,
                               self.confidence,
                               self.CUDA)
        self.optimizer = optim.Adam(self.darknet.parameters(), lr=1e-4)
        self.history = dict()
        if cfg_file[-8:-4] == 'tiny':
            self.TINY = True
        else:
            self.TINY = False
        if weights_file is not None:
            self.darknet.load_weights(weights_file)

        # using GPUs for training
        self.device = torch.device('cuda:0' if self.CUDA
                                   else 'cpu')
        self.darknet.to(self.device)
        if torch.cuda.device_count() > 1:
            self.darknet = nn.DataParallel(self.darknet)
        self.darknet = self.darknet.train()
        print("\nTrainer is ready!!\n")
        print('GPU usage = {}\n'.format(self.CUDA))

    def COCO_loader(self, json_dir, img_dir,
                    batch_size, shuffle) -> None:
        """Setting the dataloaders for the training

        Parameters:
            directory (str): Directory of the folder containing the images
            batch_size (int): Size of the mini batches
            shuffle (bool): When True, dataset images will be shuffled
        """

        assert isinstance(json_dir, str)
        assert isinstance(img_dir, str)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle, bool)
        self.dataset = COCO(json_dir, img_dir, resolution=self.resolution)
        self.data_num = self.dataset.__len__()
        self.dataloader = self.dataset.get_dataloader(batch_size=batch_size,
                                                      shuffle=shuffle)
        print('DataLoader is created successfully!\n')

    # to be continue
    def target_creator(self, bndbox):
        output = []
        anchors = self.darknet.anchors
        for bboxes in bndbox:
            layer_1 = self.target_layer(bboxes, 13, anchors[:3])
            layer_2 = self.target_layer(bboxes, 26, anchors[3:6])
            if self.TINY:
                output.append(torch.cat((layer_1, layer_2), dim=0))
            else:
                layer_3 = self.target_layer(bboxes, 52, anchors[6:])
                output.append(torch.cat((layer_1, layer_2, layer_3), dim=0))
        output = torch.stack(output, dim=0)
        return output

    def target_layer(self, bboxes: torch.Tensor, scale, anchors):
        '''Bounding boxes are in xywh format'''
        output = torch.zeros((scale*scale*len(anchors),
                              5 + self.num_classes))
        zeros = torch.zeros(5 + self.num_classes)
        stride = self.resolution//scale
        for box in bboxes:
            x_off = int(box[0].item()/stride)
            y_off = int(box[1].item()/stride)
            anchor_fit = self.anchor_fit(box[2].item()/box[3].item(), anchors)
            for i in range(len(anchor_fit)):
                loc = (x_off*scale + y_off)*len(anchors) + anchor_fit[i]
                if (output[loc] != zeros).float().sum() == 0:
                    output[loc] = box
                    output[loc, 4] = 1 - 0.1*i
        return output

    def anchor_fit(self, aspect_ratio: float, anchors: list):
        anchors_aspect_ratio = [float(x[0]/x[1]) for x in anchors]
        fits = [(x-aspect_ratio)**2 for x in anchors_aspect_ratio]
        fits_sorted = fits.copy()
        fits_sorted.sort()
        output = [fits.index(r) for r in fits_sorted]
        return output

    @staticmethod
    def progress_bar(curr_epoch, curr_batch, batch_num, loss,
                     t1=None, t2=None):
        percent = curr_batch/batch_num
        last = int((percent*1000) % 10)
        percent = round(percent*100)
        bar = 'Epoch: {:3d} '.format(curr_epoch)
        bar += 'Batch: {:3d} '.format(curr_batch)
        bar += 'Loss: {:.4f} '.format(loss)
        # bar += 'ETA: {:.2f} '.format(t2-t1)
        bar += '|' + '#' * int(percent)
        if curr_batch != batch_num:
            bar += '{}'.format(last)
            bar += ' ' * (100-int(percent)) + '|'
            print('\r'+bar, end='')
        else:
            bar += '#'
            bar += ' ' * (100-int(percent)) + '|'
            print('\r'+bar)

    def train(self, annotation_dir, img_dir):
        """Training the, batch_size = 8 network for the given dataset and network
        specifications. Batch size and epoch number must be initialized.

        Parameters:
            directory (str): Directory of the folder containing dataset images
        """

        assert isinstance(annotation_dir, str)
        assert isinstance(img_dir, str)
        # initializations for the training
        # mem_loss = 0.0
        # memory_epoch = 0
        # stop_training = False
        self.history['train_loss'] = [0]*self.epoch

        # dataloader adjustment
        '''
        self.VOC_loader(xml_directory, img_directory,
                        batch_size=self.batch_size,
                        shuffle=True)'''
        self.COCO_loader(annotation_dir, img_dir,
                         batch_size=self.batch_size,
                         shuffle=True)

        batch_num = ceil(self.data_num/self.batch_size)
        for epoch in range(1, self.epoch+1):
            running_loss = 0.0

            # training mini-batches
            for batch, batch_samples in enumerate(self.dataloader):
                samples = batch_samples[0]
                bndbox = batch_samples[1]
                if self.CUDA:
                    batch_data = samples.clone().cuda()
                else:
                    batch_data = samples.clone()
                del batch_samples, samples

                # making the optimizer gradient zero
                self.optimizer.zero_grad()
                pred = self.darknet(batch_data)
                # t1 = time.time()
                # prediction = self.activation_pass(prediction)
                target = self.target_creator(bndbox)
                if self.CUDA:
                    target = target.cuda()

                # prediction = self.conf_masking(prediction, confidence=0.6)
                # t2 = time.time()
                # prediction = self.pred_processor(prediction)
                # t3 = time.time()
                # prediction = self.row_sort(prediction)
                # t4 = time.time()
                # print(prediction)
                loss = nn.functional.mse_loss(pred, target,
                                              reduction='sum')
                # t3 = time.time()
                loss.backward()
                # t4 = time.time()
                self.optimizer.step()
                # t5 = time.time()
                # print('target time= ', t2 - t1)
                # print('loss time= ', t3 - t2)
                # print('backward time= ', t4 - t3)
                # print('step time= ', t5 - t4)
                # print('backward time= ', t6 - t5)
                # print('step time= ', t7 - t6)

                # loss at the end of the batch
                running_loss += loss.item()
                self.progress_bar(epoch, batch, batch_num, loss.item())
                del batch_data, bndbox
                torch.cuda.empty_cache()
                torch.save(self.darknet.state_dict(),
                           'weights/training_output')

            print('Epoch Loss = {}\n'.format(running_loss/(batch+1)))
            self.history['train_loss'][epoch-1] = running_loss/(batch+1)

        # when the training is finished
        print('Training is finished !!\n')
        torch.save(self.darknet.state_dict(),
                   'weights/training_output')
        epochs = [item for item in range(1, self.epoch+1)]
        plt.plot(epochs, self.history['train_loss'], color='red')
        # plt.plot(epochs, self.history['val_loss'], color='blue')
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.savefig('weights/loss_graph.png')


def arg_parse():
    """Training file argument configuration"""

    # default arguments
    ann_def_VOC = '/home/adm1n/Datasets/SPAutoencoder/\
VOCdevkit/VOC2012/Annotations'
    img_def_VOC = '/home/adm1n/Datasets/SPAutoencoder/VOC2012'
    cfg_def = 'cfg/yolov3-tiny.cfg'
    weights_def = 'weights/yolov3-tiny.weights'
    ann_def_COCO = '/home/adm1n/Datasets/\
COCO/2017/annotations/instances_val2017.json'
    img_def_COCO = '/home/adm1n/Datasets/COCO/2017/val2017/'

    # argument parsing
    parser = argparse.ArgumentParser(description='YOLO v3 Training Module')

    parser.add_argument("--xml", dest='xml',
                        help="Ground Truth directory of the training images",
                        default=ann_def_COCO, type=str)
    parser.add_argument("--images",
                        help="""Image / Directory containing
                                images to perform training upon""",
                        default=img_def_COCO, type=str)
    parser.add_argument("--batch_size", dest="bs",
                        help="Batch size of training",
                        default=64, type=int)
    parser.add_argument("--epoch", dest="epoch",
                        help="Epoch Number of training",
                        default=40, type=int)
    parser.add_argument("--confidence", dest="conf",
                        help="Object Confidence to filter predictions",
                        default=0.6, type=float)
    parser.add_argument("--cfg", dest='cfg_file', help="Config file",
                        default=cfg_def, type=str)
    parser.add_argument("--weights", dest='weights_file', help="weightsfile",
                        default=weights_def, type=str)
    parser.add_argument("--reso", dest='reso',
                        help="""Input resolution of the network. Increase to
                        increase accuracy. Decrease to increase speed""",
                        default=416, type=str)
    parser.add_argument("--use_GPU", dest='CUDA', action='store_true',
                        help="GPU Acceleration Enable Flag (true/false)")

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    xml_dir = args.xml
    img_dir = args.images
    batch_size = int(args.bs)
    epoch_number = int(args.epoch)
    confidence = float(args.conf)
    cfg = args.cfg_file
    weights = args.weights_file
    reso = int(args.reso)
    CUDA = args.CUDA
    assert type(CUDA) == bool
    trainer = DarknetTrainer(cfg, weights,
                             epoch=epoch_number,
                             batch_size=batch_size,
                             resolution=reso, confidence=confidence, CUDA=CUDA)
    trainer.train(xml_dir, img_dir)
