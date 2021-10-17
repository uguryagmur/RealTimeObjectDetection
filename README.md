# RealTimeObjectDetection
YOLOv3 and YOLOv5 implementations for real time object detections

## Weights
In order to get the weights for YOLOv3 and YOLOv3-tiny, run the bash script
provided in the system *get_weights.sh*

Weights for YOLOv5 model will be downloaded automatically when you use it 
from Pytorch Hub.

## Usage

- Fill the parameters in the *params.json* file. If you use detection, you can ignore the training parameters.
- For detection, run the command:
    > python main.py detect
- For training a YOLOv3 model, run the command:
    > python main.py train

### Parameters
#### **Detector Parameters**
| **parameter name** | definition of parameter |
| ------------------- | -----------------------|
| **images_path** | Path of the folder containing images to detect object in them |
| **destination_path** | Path of the folder detected images will be written |
| **cfg_file_path** | Confiugration file path for v3 models |
| **weights_file_path** | Weights file path for v3 models |
| **resolution** | Squared dimensions of the input images |
| **confidence** | Object confidence scores |
| **nms_threshold** |Non-max Suppression Threshold Value |
| **use_torch_weights** | Flag for using a trained pytorch weights |
| **CUDA** | Flag for enabling GPU usage (only Nvidia GPUs) |

---------------------------------------

#### Training Parameters
| **parameter name** | definition of parameter |
| ------------------- | -----------------------|
| **number_of_epoch** | Epoch number for training|
| **batch_size** | Batch size (number of images) per mini batch|
| **start_from_checkpoint** | Flag to continue a stopped training**|
| **train_img_dir** | Directory of the folder containing training images |
| **train_annot_dir** | Directory of the folder/file containing training annotations |
| **train_dataset_type** | Training dataset type: VOC or COCO |
| **valid_img_dir** | Directory of the folder containing validation images|
| **valid_annot_dir** | Directory of the folder/file containing validation annotations |
| **valid_dataset_type** | Validation dataset type: VOC or COCO |

**For this flag, trainable parameters of model and optimizer have to be provided to continue to the training

### Using *detect.py* directly
For using the *detect.py* directly, you have to give some parameters like weights and cfg file. You can see all 
explanations of the parameters by using the command:
> python detect.py --help

Example usage for YOLOv3 detection in imgs folder:
> python detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights --images imgs

### Using *train.py* directly
For using the *detect.py* directly, you have to give some parameters like weights and cfg file. You can see all 
explanations of the parameters by using the command:
> python train.py --help

For a faster and more convenient training use multiple GPUs.

### To Do List
- [X] Add main file to run the system with parameters
- [X] Update the user manual
- [ ] Add YOLOv5 support for detection
- [ ] Add YOLOv5 support for training
- [ ] Add automatic dataset configuration for COCO and VOC datasets