""" Validation for YOLOv3 Training with Precision Recall & F Scores """

import torch
import json
from src.darknet import Darknet
from src.dataset import COCO
from src.util import bbox_iou, write_results, xywh2xyxy

torch.manual_seed(7)


class DarknetValidator:
    """Darknet YOLO Network Validator Class"""

    def __init__(self, annotation_dir, img_dir, confidence=0.6,
                 num_classes=80, nms_thresh=0.5, resolution=416):
        assert isinstance(resolution, int) and resolution % 32 == 0
        assert confidence < 1 and confidence > 0
        assert nms_thresh < 1 and nms_thresh > 0
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.resolution = resolution
        self.num_classes = int(num_classes)
        self.set_dataloader(annotation_dir, img_dir)
        self.image_scores = {}
        self.total_scores = {}
        self.total_scores['people_num'] = 0
        self.total_scores['tp'] = 0
        self.total_scores['fn'] = 0
        self.total_scores['fp'] = 0

    def set_dataloader(self, annotation_dir, img_dir):
        assert isinstance(annotation_dir, str)
        assert isinstance(img_dir, str)
        self.dataset = COCO(annotation_dir, img_dir,
                            self.resolution, keep_img_name=True)
        self.data_num = self.dataset.__len__()
        self.dataloader = self.dataset.get_dataloader(batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1)

    def target_filter(self, target: torch.Tensor, permitted_classes: tuple,
                      min_box_size=0):
        output = []
        for i in range(len(target)):
            if target[i, 2] > min_box_size and target[i, 3] > min_box_size and\
               torch.argmax(target[i, 5:]).item() in permitted_classes:
                output.append(target[i].clone())
        if len(output) > 0:
            output = torch.stack(output, dim=0)
            output = xywh2xyxy(output)
            return output
        else:
            return None

    def pred_filter(self, pred: torch.Tensor, permitted_classes: tuple):
        if type(pred) == int:
            return 0
        output = []
        for i in range(len(pred)):
            if pred[i, -1] in permitted_classes:
                output.append(pred[i])
        if len(output) > 0:
            output = torch.stack(output, dim=0)
            return output
        else:
            return 0

    def compare_boxes(self, pred, target, threshold: float):
        box_ious = []
        row = []
        true_positive = 0
        false_negative = 0
        for box in pred:
            for t_box in target:
                iou = bbox_iou(box[1:5].cpu(), t_box[0:4].cpu())
                if iou.item() > threshold:
                    row.append(iou.item())
                else:
                    row.append(0.0)
            box_ious.append(row.copy())
            row.clear()

        box_ious = torch.FloatTensor(box_ious)

        # if there is an anormal change on pred tensor problem might be there
        for i in range(pred.size(0)):
            if torch.max(box_ious).item() == 0:
                break
            max_val, max_ind = torch.max(box_ious, dim=1)
            ind = torch.argmax(max_val).long()
            box_ious[ind] = torch.zeros(box_ious[ind].size())
            box_ious[:, max_ind[ind]] = torch.zeros(
                box_ious[:, max_ind[ind]].size())
            true_positive += 1

        return true_positive

    def save_img_scores_(self, img_name, people_num, tp, fp, fn):
        self.image_scores[img_name] = {'people_num': people_num,
                                       'tp': tp,
                                       'fp': fp,
                                       'fn': fn}

    def save_total_scores_(self, people_num, tp, fp, fn):
        self.total_scores['people_num'] += people_num
        self.total_scores['tp'] += tp
        self.total_scores['fp'] += fp
        self.total_scores['fn'] += fn

    def get_img_scores(self, img_name, pred, target, img_scores=False):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        people_num = 0
        if type(pred) == int and target is None:
            return
        elif type(pred) == int and target is not None:
            people_num = target.size(0)
        elif type(pred) != int and target is None:
            false_positive = pred.size(0)
        elif type(pred) != int and target is not None:
            people_num = target.size(0)
            true_positive = self.compare_boxes(pred, target, 0.4)
            false_positive = pred.size(0) - true_positive

        false_negative = people_num - true_positive
        # print(img_name, people_num, true_positive,
        #       false_positive, false_negative)

        if img_scores:
            self.save_img_scores_(img_name, people_num, true_positive,
                                  false_positive, false_negative)

        self.save_total_scores_(people_num, true_positive,
                                false_positive, false_negative)

    def save_scores(self, img_score_dir=None, total_score_dir=None):
        if img_score_dir is not None:
            json.dump(self.image_scores, open(img_score_dir, 'w'))
        if total_score_dir is not None:
            json.dump(self.total_scores, open(total_score_dir, 'w'))

    @staticmethod
    def progress_bar(curr_batch, batch_num):
        bar_length = 100
        percent = curr_batch/(batch_num-1)*100
        bar = 'Validation Process: '
        bar += '\t\t\t{:>3.2f}% '.format(percent)
        percent = round(percent)
        bar += '|' + '=' * int(percent*bar_length/100)
        if curr_batch == batch_num - 1:
            bar += ' ' * (bar_length - int(percent*bar_length/100)) + '|'
            print('\r'+bar)
        else:
            bar += '>'
            bar += ' ' * (bar_length - int(percent*bar_length/100)) + '|'
            print('\r'+bar, end='')

    def validate_model(self, model: Darknet, CUDA=False, img_scores=False):
        for batch, data in enumerate(self.dataloader):
            img_name = data[0][0]
            samples = data[1]
            bndbox = data[2][0]
            bndbox = self.target_filter(bndbox, [0], min_box_size=24)
            # draw_boxes(samples.squeeze(0), bndbox, None, from_tensor=True)
            if CUDA:
                samples = samples.cuda()
            with torch.no_grad():
                pred = model(samples)

            pred = write_results(pred, self.num_classes,
                                 self.confidence, self.nms_thresh)

            # applying filters for certain classes
            pred = self.pred_filter(pred, [0])
            self.get_img_scores(img_name, pred, bndbox, img_scores)
            self.progress_bar(batch, self.data_num)

        tp = torch.tensor(self.total_scores['tp']).float()
        fp = torch.tensor(self.total_scores['fp']).float()
        fn = torch.tensor(self.total_scores['fn']).float()
        self.precision = (tp/(tp + fp)).clone()
        self.recall = (tp/(tp + fn)).clone()
        self.f_score = (2/((1/self.recall) + (1/self.precision))).clone()
        print('\tPrecision = ', self.precision)
        print('\tRecall = ', self.recall)
        print('\tF_Score = ', self.f_score)
        # self.save_scores(img_score_dir='img_scores.json',
        #                  total_score_dir='total_scores.json')

    def validate_json(self, json_dir, img_scores=False):
        pred_dict = json.load(open(json_dir, 'r'))
        with self.dataset.only_ground_truth():
            for batch, data in enumerate(self.dataloader):
                img_name = data[0][0]
                bndbox = data[1][0]
                pred = torch.FloatTensor(pred_dict[img_name])
                bndbox = self.target_filter(bndbox, [0], min_box_size=24)
                pred = self.pred_filter(pred, [0])
                self.get_img_scores(img_name, pred, bndbox, img_scores=True)
                self.progress_bar(batch, self.data_num)

        tp = torch.tensor(self.total_scores['tp']).float()
        fp = torch.tensor(self.total_scores['fp']).float()
        fn = torch.tensor(self.total_scores['fn']).float()
        self.precision = (tp/(tp + fp)).clone()
        self.recall = (tp/(tp + fn)).clone()
        self.f_score = (2/((1/self.recall) + (1/self.precision))).clone()
        print('\tPrecision = ', self.precision)
        print('\tRecall = ', self.recall)
        print('\tF_Score = ', self.f_score)
        self.save_scores(img_score_dir='img_scores.json',
                         total_score_dir='total_scores.json')


if __name__ == '__main__':
    cfg_file = 'cfg/yolov3.cfg'
    weights_file = 'weights/yolov3.weights'
    annot_dir = '/home/adm1n/Datasets/COCO/2017/annotations\
/instances_val2017.json'
    img_dir = '/home/adm1n/Datasets/COCO/2017/val2017/'
    model = Darknet(cfg_file, CUDA=True).cuda()
    model.load_weights(weights_file)
    # model.load_state_dict(torch.load('weights/experiment3/checkpoint8'))
    validator = DarknetValidator(annot_dir, img_dir)
    validator.validate_model(model, CUDA=True, img_scores=True)
    json_dir = 'metrics.json'
    # validator.validate_json(json_dir, img_scores=True)
