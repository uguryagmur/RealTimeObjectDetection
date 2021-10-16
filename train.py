"""YOLOv3 Darknet Trainer of the Network"""

import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from test import DarknetValidator
from src.dataset import COCO, VOC
from src.darknet import Darknet
from src.util import xywh2YOLO, bbox_iou_wh

torch.manual_seed(42)


class DarknetTrainer:
    """Darknet YOLO Network Trainer Class

    Attributes:
        epoch (int): Epoch number of the training
        batch_size (int): Size of the mini-batches
        dataset (COCO): Dataset to train network
        train_loader (DataLoader): torch DataLoader object for training set
        darknet (Darknet): Darknet network to train
        optimizer (torch.optim): Optimizer to train network
        criterion (torch.nn.MSELoss): Criterion for the loss of network output
        device (torch.device): Device running the training process
        validator (None, DarknetValidator): object for validation if it is not None
        history (dict): loss and metrics history for the training
        TINY (bool): flag to decide whether the Darknet is tiny one
    """

    def __init__(self, cfg_file: str, weights_file=None,
                 epoch=10, batch_size=16, resolution=416,
                 confidence=0.6, num_classes=80, patience=3, CUDA=False,
                 TUNE=False) -> None:
        """ Constructor of the Darknet Trainer Class """

        assert isinstance(epoch, int)
        assert isinstance(batch_size, int)
        assert isinstance(resolution, int)
        assert resolution % 32 == 0
        self.CUDA = bool(torch.cuda.is_available() and CUDA)
        self.num_classes = num_classes
        self.epoch = epoch
        self.patience = patience
        self.batch_size = batch_size
        self.resolution = resolution
        self.confidence = confidence
        self.criterion = self.darknet_loss
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.BCELoss = nn.BCELoss(reduction='sum')
        self.darknet = Darknet(cfg_file,
                               self.CUDA)
        self.darknet.net_info["height"] = resolution
        self.optimizer = optim.Adam(self.darknet.parameters(), lr=1e-2)
        self.validator = None
        self.history = dict()
        self.configure_for_tiny_model(cfg_file)
        self.load_weights(weights_file)
        self.configure_darknet(TUNE)
        print("\nTrainer is ready!!\n")
        print('GPU usage = {}\n'.format(self.CUDA))

    def configure_darknet(self, TUNE):
        # using GPUs for training
        self.device = torch.device('cuda:0' if self.CUDA
                                   else 'cpu')
        self.darknet.to(self.device)
        if torch.cuda.device_count() > 1:
            self.darknet = nn.DataParallel(self.darknet)
        if TUNE:
            self.darknet.load_state_dict(torch.load('weights/training_output'))
            self.optimizer.load_state_dict(
                torch.load('weights/training_output_opt'))
        self.darknet = self.darknet.train()

    def load_weights(self, weights_file):
        if weights_file is not None:
            self.darknet.load_weights(weights_file)

    def configure_for_tiny_model(self, cfg_file):
        if cfg_file[-8:-4] == 'tiny':
            self.TINY = True
        else:
            self.TINY = False

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

    def VOC_loader(self, xml_dir, img_dir,
                   batch_size, shuffle) -> None:
        """Setting the dataloaders for the training

        Parameters:
            directory (str): Directory of the folder containing the images
            batch_size (int): Size of the mini batches
            shuffle (bool): When True, dataset images will be shuffled
        """

        assert isinstance(xml_dir, str)
        assert isinstance(img_dir, str)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle, bool)
        self.dataset = VOC(xml_dir, img_dir, resolution=self.resolution)
        self.data_num = self.dataset.__len__()
        self.dataloader = self.dataset.get_dataloader(batch_size=batch_size,
                                                      shuffle=shuffle)
        print('DataLoader is created successfully!\n')

    def target_creator(self, bndbox):
        """
        Creating the target output for the Darknet network

        Arguments:
            bndbox (torch.Tensor): bounding boxes of the batch to create
                target tensor
        """
        output = []
        mask = []
        anchors = self.darknet.anchors

        for i in range(len(bndbox)):
            layer_1, mask_1 = self.target_layer(bndbox[i], 13, anchors[:3])
            layer_2, mask_2 = self.target_layer(bndbox[i], 26, anchors[3:6])
            if self.TINY:
                self.create_target_layer_for_tiny(layer_1, layer_2, mask, mask_1, mask_2, output)
            else:
                self.create_target_layer_for_main(anchors, bndbox, i, layer_1, layer_2, mask, mask_1, mask_2, output)
        mask, output = self.stack_output_and_mask(mask, output)
        return output, mask

    @staticmethod
    def stack_output_and_mask(mask, output):
        output = torch.stack(output, dim=0)
        mask = torch.stack(mask, dim=0).bool()
        return mask, output

    def create_target_layer_for_main(self, anchors, bndbox, i, layer_1, layer_2, mask, mask_1, mask_2, output):
        layer_3, mask_3 = self.target_layer(bndbox[i], 52, anchors[6:])
        output.append(torch.cat((layer_1, layer_2, layer_3), dim=0))
        mask.append(torch.cat((mask_1, mask_2, mask_3), dim=0))

    @staticmethod
    def create_target_layer_for_tiny(layer_1, layer_2, mask, mask_1, mask_2, output):
        output.append(torch.cat((layer_1, layer_2), dim=0))
        mask.append(torch.cat((mask_1, mask_2), dim=0))

    def target_layer(self, bboxes: torch.Tensor, scale, anchors):
        """Function to create target for layer with respect to given scale

        Parameters:
            bboxes (torch.Tensor): bounding box tensors
            scale (int): scale of the corresponding detection layer
            anchors (list): list of tuples of the anchor box pairs
        """
        # Bounding boxes are in xywh format
        output = torch.zeros((scale*scale*len(anchors),
                              5 + self.num_classes))
        mask = torch.zeros(output.shape[:-1])
        stride = self.resolution//scale
        for box in bboxes:
            if box[5] != 1:
                continue
            elif box[2] < 24 or box[3] < 24:
                continue
            anchor_fit = self.anchor_fit(box[:4], anchors)
            best_anchor = anchors[anchor_fit]
            w_coor, h_coor, w_center, h_center, w, h = xywh2YOLO(box, stride,
                                                                 best_anchor)
            loc = (w_coor*scale + h_coor)*len(anchors) + anchor_fit
            output[loc] = box
            output[loc][:4] = torch.FloatTensor([w_center, h_center, w, h])
            mask[loc] = 1
        return output, mask

    @staticmethod
    def anchor_fit(box: torch.Tensor, anchors: list):
        """Function to return best fitting anchor box to the corresponding
        bounding box

        Parameters:
            box (torch.Tensor): bounding box tensor
            anchors (list): list of tuples of anchors box width-height
        """
        output = []
        w_box, h_box = box[2].item(), box[3].item()
        for i in range(len(anchors)):
            output.append(bbox_iou_wh((w_box, h_box), anchors[i]))
        output = output.index(max(output))
        return output

    def darknet_loss(self, pred, target, obj_mask):
        """Function to calculate loss of Darknet

        Parameters:
            pred (torch.Tensor): prediction output of the network
            target (torch.Tensor): target of the network
            obj_mask (torch.Tensor): object mask for the target
        """
        no_obj_mask = (torch.ones(obj_mask.size()) - obj_mask.float()).bool()
        loss = 5*self.MSELoss(pred[obj_mask][..., :2],
                              target[obj_mask][..., :2])
        loss += 5*self.MSELoss(pred[obj_mask][..., 2:4],
                               target[obj_mask][..., 2:4])
        loss += 1*self.MSELoss(pred[obj_mask][..., 4],
                               target[obj_mask][..., 4])
        loss += 0.5*self.MSELoss(pred[no_obj_mask][..., 4],
                                 target[no_obj_mask][..., 4])
        loss += self.MSELoss(pred[obj_mask][..., 5:],
                             target[obj_mask][..., 5:])
        return loss

    def get_validator(self, annotation_dir, img_dir):
        """Function to initialize validator object

        Parameters:
            annotation_dir (str): annotation directory of the validation
            img_dir (str): path of the folder containing validation images
        """
        self.validator = DarknetValidator(annotation_dir,
                                          img_dir, confidence=self.confidence)

    @staticmethod
    def progress_bar(curr_epoch, epoch_num, curr_batch, batch_num, loss):
        """This function shows the progress bar of the training

        Arguments:
            curr_epoch (int): current epoch number
            epoch_num (int): total epoch number
            curr_batch (int): current batch number
            batch_num (int): total batch number
            loss (float): loss of the current batch
        """
        bar_length = 100
        percent = curr_batch/batch_num*100
        bar = 'Epoch: {:3d} '.format(curr_epoch)
        bar += 'Batch: {:3d} '.format(curr_batch)
        bar += 'Loss: {:<8.2f}\t'.format(loss)
        bar += '{:>3.2f}% '.format(percent)
        percent = round(percent)
        bar += '|' + '=' * int(percent*bar_length/100)
        if curr_batch == batch_num:
            bar += ' ' * (bar_length - int(percent*bar_length/100)) + '|\n'
            print('\r'+bar)
        else:
            bar += '>'
            bar += ' ' * (bar_length - int(percent*bar_length/100)) + '|'
            print('\r'+bar, end='')

    @staticmethod
    def epoch_ETA(*args, remaining_epoch) -> None:
        """
        Function to print the estimated time arrival for the training

        args (list): start and end times of the processes in training
        remaining_epoch (int): number of remaining epoch
        """
        assert len(args) == 2
        delta = sum([args[i] - args[i+1] for i in range(0, len(args), 2)])
        delta *= remaining_epoch
        ETA_h = int(delta/3600)
        ETA_m = int((delta % 3600)/60)
        ETA_s = int((delta % 3600) % 60)
        print('\tETA: {0}:{1}:{2}\n'.format(ETA_h, ETA_m, ETA_s))

    @staticmethod
    def epoch_loss(loss, batch_data_length) -> None:
        """
        Function to print loss

        Parameters:
            loss (float): Loss of the current epoch
            batch_data_length (int): number of batch in one epoch
        """
        avg_loss = loss/batch_data_length
        print('\n\tAverage Epoch Loss: {}'.format(avg_loss))

    def train(self, annotation_dir, img_dir):
        """Training the, batch_size = 8 network for the given dataset and network
        specifications. Batch size and epoch number must be initialized.

        Parameters:
            annotation_dir (str): Path of the annotation for the training
            img_dir (str): Directory of the folder containing dataset images
        """
        self.initialize_training()
        best_metric = None

        # dataloader adjustment
        self.COCO_loader(annotation_dir, img_dir,
                         batch_size=self.batch_size,
                         shuffle=True)

        batch_num = self.data_num//self.batch_size + 1
        for epoch in range(1, self.epoch+1):
            running_loss, t_start = self.forward_epoch(batch_num, best_metric, epoch)
            self.update_epoch_info(batch_num, epoch, running_loss, t_start)

        # when the training is finished
        self.save_end_checkpoint_and_optimizer()
        epochs = [item for item in range(1, len(self.history['train_loss'])+1)]

        self.plot_loss_graph(epochs)
        if self.validator is not None:
            self.plot_validation_graph(epochs)
        print('Training is finished !!\n')

    def initialize_training(self):
        self.history['train_loss'] = []
        self.history['train_precision'] = []
        self.history['train_recall'] = []
        self.history['train_f_score'] = []

    def forward_epoch(self, batch_num, best_metric, epoch):
        running_loss = 0.0
        t_start = time.time()
        # training mini-batches
        for batch, batch_samples in enumerate(self.dataloader):
            running_loss = self.forward_batch(batch, batch_num, batch_samples, epoch, running_loss)

        torch.save(self.darknet.state_dict(),
                   'weights/weight_epoch' + str(epoch))

        if self.validator is None:
            best_metric = self.save_checkpoint(best_metric, running_loss)
        else:
            self.update_scores()
            best_metric = self.save_checkpoint(best_metric, self.validator.f_score)
        return running_loss, t_start

    def forward_batch(self, batch, batch_num, batch_samples, epoch, running_loss):
        samples = batch_samples[0]
        bndbox = batch_samples[1]
        batch_data = self.move_data_to_device(samples)
        del batch_samples, samples
        loss = self.feed_forward_through_model(batch_data, bndbox)
        # loss at the end of the batch
        running_loss = self.update_batch_info(batch, batch_num, epoch, loss, running_loss)
        return running_loss

    def plot_validation_graph(self, epochs):
        plt.plot(epochs, self.history['train_precision'], color='blue')
        plt.plot(epochs, self.history['train_recall'], color='green')
        plt.plot(epochs, self.history['train_f_score'], color='yellow')
        plt.legend(['precision', 'recall', 'f_score'])
        plt.xlabel('epoch number')
        plt.ylabel('metrics')
        plt.savefig('weights/metric_graph.png')

    def plot_loss_graph(self, epochs):
        plt.plot(epochs, self.history['train_loss'], color='red')
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.savefig('weights/loss_graph.png')
        plt.clf()

    def save_end_checkpoint_and_optimizer(self):
        torch.save(self.darknet.state_dict(),
                   'weights/training_output')
        torch.save(self.optimizer.state_dict(),
                   'weights/training_output_opt')

    def update_epoch_info(self, batch_num, epoch, running_loss, t_start):
        # printing the metrics of the epoch
        t_end = time.time()
        self.epoch_loss(running_loss, self.dataset.__len__())
        self.epoch_ETA(t_start, t_end, remaining_epoch=(self.epoch - epoch))
        self.history['train_loss'].append(running_loss / (batch_num))

    def update_batch_info(self, batch, batch_num, epoch, loss, running_loss):
        running_loss += loss.item()
        self.progress_bar(epoch, self.epoch, batch + 1,
                          batch_num, loss.item())
        torch.cuda.empty_cache()
        return running_loss

    def update_scores(self):
        self.validator.validate_model(self.darknet, CUDA=self.CUDA)
        self.history['train_precision'].append(
            self.validator.precision)
        self.history['train_recall'].append(self.validator.recall)
        self.history['train_f_score'].append(self.validator.f_score)

    def save_checkpoint(self, best_metric, running_loss):
        if best_metric is None or running_loss > best_metric:
            best_metric = running_loss
            torch.save(self.darknet.state_dict(),
                       'weights/checkpoint')
            torch.save(self.optimizer.state_dict(),
                       'weights/checkpoint_opt')
        return best_metric

    def feed_forward_through_model(self, batch_data, bndbox):
        # making the optimizer gradient zero
        self.optimizer.zero_grad()
        with self.darknet.train_mode():
            pred = self.darknet(batch_data)
        # creating the target of the training
        target, mask = self.target_creator(bndbox)
        if self.CUDA:
            target = target.cuda()
        # calculating the loss and backpropagation
        loss = self.criterion(pred, target, mask)
        loss.backward()
        self.optimizer.step()
        return loss

    def move_data_to_device(self, samples):
        if self.CUDA:
            batch_data = samples.clone().cuda()
        else:
            batch_data = samples.clone()
        return batch_data


def arg_parse():
    """Training file argument configuration"""

    # default arguments
    ann_def_VOC = '/home/adm1n/Datasets/SPAutoencoder/\
VOCdevkit/VOC2012/Annotations'
    img_def_VOC = '/home/adm1n/Datasets/SPAutoencoder/VOC2012'
    cfg_def = 'cfg/yolov3-tiny.cfg'
    weights_def = 'weights/yolov3-tiny.weights'
    ann_def_COCO = '/home/adm1n/Datasets/COCO/2017\
/annotations/instances_val2017.json'
    img_def_COCO = '/home/adm1n/Datasets/COCO/2017/val2017'

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
                        default=16, type=int)
    parser.add_argument("--epoch", dest="epoch",
                        help="Epoch Number of training",
                        default=30, type=int)
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
    parser.add_argument("--fine_tune", dest='TUNE', action='store_true',
                        help="Fine Tune for last output Flag (true/false)")

    return parser.parse_args()


if __name__ == '__main__':
    params = arg_parse()
    trainer_params = {
        "cfg_file": params.cfg_file,
        "weights_file": params.weights_file,
        "epoch": params.epoch,
        "batch_size": params.bs,
        "resolution": params.reso,
        "confidence": params.conf,
        "CUDA": params.CUDA,
        "TUNE": params.TUNE,
    }
    validator_params = {
        "annotation_dir": '/home/adm1n/Datasets/COCO/2017/annotations/instances_val2017.json',
        "img_dir": '/home/adm1n/Datasets/COCO/2017/val2017/',
    }
    train_params = {
        "annotation_dir": params.xml,
        "img_dir": params.images,
    }
    trainer = DarknetTrainer(**trainer_params)
    trainer.get_validator(**validator_params)
    trainer.train(**train_params)
