"""YOLOv3 Darknet Trainer of the Network"""

# import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.darknet import Darknet
from src.dataloader import VOCDataset


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
                 confidence=0.6, CUDA=False) -> None:
        """ Constructor of the Darknet Trainer Class """

        assert isinstance(weights_file, (str, None))
        assert isinstance(epoch, int)
        assert isinstance(batch_size, int)
        assert isinstance(resolution, int)
        assert resolution % 32 == 0
        self.CUDA = bool(torch.cuda.is_available() and CUDA)
        self.epoch = epoch
        self.batch_size = batch_size
        self.resolution = resolution
        self.criterion = self.darknet_loss
        self.darknet = Darknet(cfg_file, CUDA=self.CUDA)
        self.optimizer = optim.Adam(self.darknet.parameters())
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

    def set_dataloader(self, xml_directory, img_directory,
                       batch_size, shuffle) -> None:
        """Setting the dataloaders for the training

        Parameters:
            directory (str): Directory of the folder containing the images
            batch_size (int): Size of the mini batches
            shuffle (bool): When True, dataset images will be shuffled
        """

        assert isinstance(xml_directory, str)
        assert isinstance(img_directory, str)
        assert isinstance(batch_size, int)
        assert isinstance(shuffle, bool)
        self.dataset = VOCDataset(xml_directory,
                                  img_directory, resolution=self.resolution)
        self.dataset.random_spilt()
        self.train_loader,\
            self.val_loader = self.dataset.get_loader(batch_size=batch_size)
        print('Train and Validation DataLoaders are created successfully!\n')

    @staticmethod
    def conf_masking(prediction: torch.Tensor,
                     confidence: float) -> torch.Tensor:
        prediction[:, :, 2:4] = prediction[:, :, 2:4].sqrt()
        conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
        prediction *= conf_mask
        return prediction

    def pred_processor(self, prediction: torch.Tensor) -> torch.Tensor:
        maximum, ind = torch.max(prediction[:, :, 5:], dim=2, keepdim=True)
        prediction = torch.cat((prediction[:, :, :5], maximum,
                                ind.float()), dim=2)
        return prediction

    @staticmethod
    def row_sort(X: torch.Tensor, descending=True) -> torch.Tensor:
        X = X[torch.arange(X.shape[0]).reshape(X.shape[0], 1).repeat(
            1, X.shape[1]), X[:, :, 0].argsort(dim=1, descending=True)]
        return X

    def darknet_loss(self, prediction: torch.Tensor,
                     target: tuple) -> torch.Tensor:
        target_ = []
        for i in range(len(target)):
            num_boxes = target[i].shape[0]
            patch = torch.tensor([[1, 1, 0]] * num_boxes).float()
            temp_ = target[i]
            temp = temp_.new(temp_.shape).float()
            temp[:, 0] = (temp_[:, 0] + temp_[:, 2])/2
            temp[:, 1] = (temp_[:, 1] + temp_[:, 2])/2
            temp[:, 2] = (temp_[:, 2] - temp_[:, 0]).float().sqrt()
            temp[:, 3] = (temp_[:, 3] - temp_[:, 1]).float().sqrt()
            temp[2:4] = temp[2:4].sqrt()
            temp = torch.cat((temp, patch), dim=1)
            if self.TINY:
                zero_matrix = torch.zeros((1, 2535-num_boxes, 7))
            else:
                zero_matrix = torch.zeros((1, 10647-num_boxes, 7))
            temp = temp.unsqueeze(dim=0)
            temp = torch.cat((temp, zero_matrix), dim=1)
            target_.append(temp)
        target_ = torch.stack(target_).squeeze(dim=1)
        if self.CUDA:
            loss = nn.functional.mse_loss(prediction,
                                          target_.cuda(), reduction='sum')
        else:
            loss = nn.functional.mse_loss(prediction, target_, reduction='sum')
        return loss

    def train(self, xml_directory, img_directory):
        """Training the, batch_size = 8 network for the given dataset and network
        specifications. Batch size and epoch number must be initialized.

        Parameters:
            directory (str): Directory of the folder containing dataset images
        """

        assert isinstance(xml_directory, str)
        assert isinstance(img_directory, str)
        # initializations for the training
        mem_loss = 0.0
        memory_epoch = 0
        stop_training = False
        self.history['train_loss'] = [0]*self.epoch
        self.history['val_loss'] = [0]*self.epoch

        # dataloader adjustment
        self.set_dataloader(xml_directory, img_directory,
                            batch_size=self.batch_size,
                            shuffle=True)

        for epoch in range(1, self.epoch+1):
            running_loss = 0.0

            # training mini-batches
            for batch, batch_samples in enumerate(self.train_loader):
                samples = batch_samples[0]
                bndbox = batch_samples[1]
                if self.CUDA:
                    batch_data = samples.clone().cuda()
                else:
                    batch_data = samples.clone()
                del batch_samples, samples

                # making the optimizer gradient zero
                self.optimizer.zero_grad()
                prediction = self.darknet(batch_data)
                # t1 = time.time()
                prediction = self.conf_masking(prediction, confidence=0.6)
                # t2 = time.time()
                prediction = self.pred_processor(prediction)
                # t3 = time.time()
                prediction = self.row_sort(prediction)
                # t4 = time.time()
                loss = self.criterion(prediction, bndbox)
                # t5 = time.time()
                loss.backward()
                # t6 = time.time()
                self.optimizer.step()
                # t7 = time.time()
                # print('masking time= ', t2 - t1)
                # print('processor time= ', t3 - t2)
                # print('sort time= ', t4 - t3)
                # print('loss time= ', t5 - t4)
                # print('backward time= ', t6 - t5)
                # print('step time= ', t7 - t6)

                # loss at the end of the batch
                running_loss += loss.item()
                print('Epoch number = {0} batch_number = {1}\n'.format(
                    epoch, batch+1))
                print('\tdarknet loss = {:.6f}\n'.format(running_loss /
                                                         (batch+1)))
                del batch_data, bndbox
                torch.cuda.empty_cache()

            print('Epoch Loss = {}\n'.format(running_loss/(batch+1)))
            self.history['train_loss'][epoch-1] = running_loss/(batch+1)

            # validation process
            val_loss = 0.0
            for batch, valid_samples in enumerate(self.val_loader):
                samples = valid_samples[0]
                bndbox = valid_samples[1]
                if self.CUDA:
                    valid_data = samples.clone().cuda()
                else:
                    valid_data = samples.clone()
                del valid_samples, samples

                with torch.no_grad():
                    val_out = self.darknet(valid_data)
                    val_out = self.conf_masking(val_out, confidence=0.6)
                    val_out = self.pred_processor(val_out)
                    val_out = self.row_sort(val_out)
                    loss = self.criterion(val_out, bndbox)
                    val_loss += loss.item()
            print('Epoch number = {0} batch_number = {1}\n'.format(
                epoch, batch+1))
            print('\tdarknet loss = {:.6f}\n'.format(val_loss/(batch+1)))
            self.history['val_loss'][epoch-1] = (val_loss/(batch+1))
            del valid_data, bndbox

            # validation check
            if mem_loss == 0 and epoch <= 1:
                mem_loss = val_loss  # saved as sum of mini-batch losses
                memory_epoch = epoch
                torch.save(self.darknet.state_dict(),
                           'weights/training_checkpoint')
                print('Validation Checkpoint is created\n')
            elif val_loss < mem_loss:
                mem_loss = val_loss  # saved as sum of mini-batch losses
                memory_epoch = epoch
                torch.save(self.darknet.state_dict(),
                           'weights/training_checkpoint')
                print('Validation Checkpoint is saved\n')
            elif memory_epoch + 10 < epoch:
                stop_training = True

            print('Validation Loss = {:.6f}\n'.format(val_loss))
            if stop_training:
                print('Validation Loss is not decreasing, use Checkpoint\n')
                print('Training is terminated!\n')
                break

        # when the training is finished
        print('Training is finished !!\n')
        print('Best checkpoint loss = {:.6f}\n'.format(
            mem_loss/(batch+1)))
        print('Last validation loss = {:.6f}\n'.format(
            mem_loss/(batch+1)))
        torch.save(self.darknet.state_dict(),
                   'weights/training_output')
        epochs = [item for item in range(1, self.epoch+1)]
        plt.plot(epochs, self.history['train_loss'], color='red')
        plt.plot(epochs, self.history['val_loss'], color='blue')
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.savefig('weights/loss_graph.png')


def arg_parse():
    """Training file argument configuration"""

    # default arguments
    xml_def = '/home/adm1n/Datasets/SPAutoencoder/\
VOCdevkit/VOC2012/Annotations'
    img_def = '/home/adm1n/Datasets/SPAutoencoder/VOC2012'
    cfg_def = 'cfg/yolov3.cfg'
    weights_def = 'weights/yolov3.weights'

    # argument parsing
    parser = argparse.ArgumentParser(description='YOLO v3 Training Module')

    parser.add_argument("--xml", dest='xml',
                        help="Ground Truth directory of the training images",
                        default=xml_def, type=str)
    parser.add_argument("--images",
                        help="""Image / Directory containing
                                images to perform training upon""",
                        default=img_def, type=str)
    parser.add_argument("--batch_size", dest="bs",
                        help="Batch size of training",
                        default=32, type=int)
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
