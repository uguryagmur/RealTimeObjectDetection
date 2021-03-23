# RealTimeObjectDetection
Different implementations of the Real Time Object Detection Methods

#### detect.py arguments:

In order to get help, please enter the command:

> > python detect.py --help

1. **--images** _"path of the folder which contains the images to detect objects"_ (default = imgs)  
2. **--det** _"path of the folder where the detected images will be written"_ (default = det)
3. **--bs** _"batch size, i.e. how many images will be detected simultaneously"_ (default = 1)
4. **--confidence** _"Object Confidence for filtering"_ (default = 0.5)
5. **--nms_thresh** _"Non Max Suppression Threshold"_ (default = 0.4)
6. **--cfg** _"Configure file path"_ (default = 'cfg/yolov3.cfg')
7. **--weights** _"Weights file path"_ (default = 'weights/yolov3.weights')
8. **--reso** _"Input resolution of the network. To increase the speed of the network, decrease the resolution"_ (default = 416)