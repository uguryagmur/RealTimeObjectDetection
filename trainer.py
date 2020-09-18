"""YOLOv3 Darknet Trainer of the Network"""

import numpy as np
from PIL import Image
from PIL import ImageDraw
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from src.dataset import COCO
import matplotlib.pyplot as plt
from src.darknet import Darknet
from src.util import xywh2YOLOlayer, xywh2xyxy, bbox_iou


class DarknetTrainer:
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
                 confidence=0.6, num_classes=80, CUDA=False,
                 TUNE=False) -> None:
        """ Constructor of the Darknet Trainer Class """

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
        self.criterion = nn.MSELoss(reduction='sum')
        self.conf_criterion = nn.BCELoss(reduction='sum')
        self.darknet = Darknet(cfg_file,
                               self.confidence,
                               self.CUDA,
                               TRAIN=True)
        self.darknet.net_info["height"] = resolution
        self.optimizer = optim.Adam(self.darknet.parameters(), lr=0.01)
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
        if TUNE:
            self.darknet.load_state_dict(torch.load('weights/checkpoint'))
            self.optimizer.load_state_dict(
                torch.load('weights/checkpoint_opt'))
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
    # prediction will be deleted
    def target_creator(self, bndbox):
        output = []
        mask = []
        anchors = self.darknet.anchors
        for i in range(len(bndbox)):
            layer_1, mask_1 = self.target_layer(bndbox[i], 13, anchors[:3])
            layer_2, mask_2 = self.target_layer(bndbox[i], 26, anchors[3:6])
            if self.TINY:
                output.append(torch.cat((layer_1, layer_2), dim=0))
                mask.append(torch.cat((mask_1, mask_2), dim=0))
            else:
                layer_3, mask_3 = self.target_layer(bndbox[i], 52, anchors[6:])
                output.append(torch.cat((layer_1, layer_2, layer_3), dim=0))
                mask.append(torch.cat((mask_1, mask_2, mask_3), dim=0))
        output = torch.stack(output, dim=0)
        mask = torch.stack(mask, dim=0).bool()
        return output, mask

    # prediction will be deleted
    def target_layer(self, bboxes: torch.Tensor, scale, anchors):
        '''Bounding boxes are in xywh format'''
        output = torch.zeros((scale*scale*len(anchors),
                              5 + self.num_classes))
        mask = torch.zeros(output.shape[:-1])
        # zeros = torch.zeros(5 + self.num_classes)
        stride = self.resolution//scale
        for box in bboxes:
            if box[5] != 1:
                continue
            anchor_fit = self.anchor_fit(box[:4], anchors)
            best_anchor = anchors[anchor_fit]
            # print('ANALYSIS')
            # print(stride)
            # print(box[:5])
            x, y, x_, y_, w_, h_ = xywh2YOLOlayer(box, stride,
                                                  best_anchor)
            # print(x, y, x_, y_, w_, h_)
            loc = (x*scale + y)*len(anchors) + anchor_fit
            # if (output[loc] != zeros).float().sum() == 0:
            output[loc] = box
            output[loc][:4] = torch.FloatTensor([x_, y_, w_, h_])
            mask[loc] = 1
        return output, mask

    def anchor_fit(self, box: torch.Tensor, anchors: list):
        anch_box = []
        output = []
        anch_box.append(torch.cat((box[:2], torch.FloatTensor(anchors[0])),
                                  dim=0))
        anch_box.append(torch.cat((box[:2], torch.FloatTensor(anchors[1])),
                                  dim=0))
        anch_box.append(torch.cat((box[:2], torch.FloatTensor(anchors[2])),
                                  dim=0))
        box = xywh2xyxy(box)
        for i in range(len(anch_box)):
            output.append(bbox_iou(box, xywh2xyxy(anch_box[i])).item())
        output = output.index(max(output))
        return output

    def result_loss(self, target, prediction, loss):
        if len(target) > len(prediction):
            control = target
            check = prediction
        else:
            control = prediction
            check = target
        index = 0
        for i in range(len(control)):
            if index >= len(check) or check[index, 0] > control[i, 0]:
                loss += (control[i]**2).sum()
            elif control[i, 0] > check[index, 0]:
                loss += (check[index]**2).sum()
                index += 1
            else:
                if control[i, -1] == check[index, -1]:
                    loss += ((control[i] - check[index])**2).sum()
                    index += 1
                elif check[index, -1] > control[i, -1]:
                    loss += (check[index]**2).sum()
                    index += 1
                else:
                    loss += (control[i]**2).sum()
        return loss

    @staticmethod
    def progress_bar(curr_epoch, curr_batch, batch_num, loss,
                     t1=None, t2=None):
        percent = curr_batch/batch_num
        last = int((percent*1000) % 10)
        percent = round(percent*100)
        bar = 'Epoch: {:3d} '.format(curr_epoch)
        bar += 'Batch: {:3d} '.format(curr_batch)
        bar += 'Loss: {:11.2f} '.format(loss)
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
        best_loss = None

        # dataloader adjustment
        '''
        self.VOC_loader(xml_directory, img_directory,
                        batch_size=self.batch_size,
                        shuffle=True)'''
        self.COCO_loader(annotation_dir, img_dir,
                         batch_size=self.batch_size,
                         shuffle=True)

        batch_num = self.data_num//self.batch_size + 1
        for epoch in range(1, self.epoch+1):
            running_loss = 0.0

            # training mini-batches
            for batch, batch_samples in enumerate(self.dataloader):
                samples = batch_samples[0]
                bndbox = batch_samples[1]
                # img = samples[0]
                # bbox = bndbox[0]
                # img = img.transpose(0, 1).transpose(1, 2).numpy()*255
                # img = Image.fromarray(np.uint8(img))
                # draw = ImageDraw.Draw(img)
                # for b in bbox:
                #     if b[5] != 1:
                #         continue
                #     box = b[:4].numpy()
                #     bbox = [0, 0, 0, 0]
                #     bbox[0] = int(box[0] - box[2]/2)
                #     bbox[1] = int(box[1] - box[3]/2)
                #     bbox[2] = int(box[0] + box[2]/2)
                #     bbox[3] = int(box[1] + box[3]/2)
                #     draw.rectangle(bbox, outline='red')
                # img.show()
                # exit()

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
                target, mask = self.target_creator(bndbox)
                if self.CUDA:
                    target = target.cuda()

                # print('ANALYSIS')
                # print(target[..., :5][torch.nonzero(target[..., :5], as_tuple=True)])
                # print(pred[..., :5][torch.nonzero(target[..., :5], as_tuple=True)])
                # shit = target-pred
                # print(shit[..., :5][torch.nonzero(target[..., :5], as_tuple=True)])
                # exit()

                pred[2:4] = (pred[2:4])/2
                target[2:4] = (target[2:4])/2
                box_loss = 5*self.criterion(pred[mask][:4], target[mask][:4])
                cls_loss = self.criterion(pred[mask][5:], target[mask][5:])
                pred = pred[..., 4]
                target = target[..., 4]
                noobj_mask = (torch.ones(mask.shape) - mask.float()).bool()
                conf_loss = self.conf_criterion(pred[mask], target[mask])
                conf_loss += 0.5*self.conf_criterion(pred[noobj_mask],
                                                     target[noobj_mask])

                loss = box_loss + conf_loss + cls_loss
                loss.backward()
                self.optimizer.step()

                # loss at the end of the batch
                running_loss += loss.item()
                self.progress_bar(epoch, batch+1, batch_num, loss.item())

                torch.cuda.empty_cache()
                if best_loss is None or running_loss < best_loss:
                    best_loss = running_loss
                    torch.save(self.darknet.state_dict(),
                               'weights/checkpoint')
                    torch.save(self.optimizer.state_dict(),
                               'weights/checkpoint_opt')
                del batch_data, bndbox, pred, target, loss

            print('\n\tTotal Epoch Loss = {}\n'.format(running_loss))
            self.history['train_loss'][epoch-1] = running_loss/(batch_num)

        # when the training is finished
        print('Training is finished !!\n')
        torch.save(self.darknet.state_dict(),
                   'weights/training_output')
        torch.save(self.optimizer.state_dict(),
                   'weights/training_output_opt')
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
    parser.add_argument("--fine_tune", dest='TUNE', action='store_true',
                        help="Fine Tune for last output Flag (true/false)")

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
    TUNE = args.TUNE
    assert type(CUDA) == bool
    trainer = DarknetTrainer(cfg, None,
                             epoch=epoch_number,
                             batch_size=batch_size,
                             resolution=reso, confidence=confidence,
                             CUDA=CUDA, TUNE=TUNE)
    trainer.train(xml_dir, img_dir)
