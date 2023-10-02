from enum import Enum

from motrackers.detectors import YOLOv3, Caffe_SSDMobileNet, TF_SSDMobileNetV2

class TrackerType(Enum):
    CentroidTracker = 'CentroidTracker'
    CentroidKF_Tracker = 'CentroidKF_Tracker'
    SORT = 'SORT'
    IOUTracker = 'IOUTracker'

class DetectorType(Enum):
    YOLOv3 = 'YOLOv3'
    Caffe_SSDMobileNet = 'Caffe_SSDMobileNet'
    # TF_SSDMobileNetV2 seems to draw a bounding box for "person" all around the image
    TF_SSDMobileNetV2 = 'TF_SSDMobileNetV2'
    
    @property
    def detector_class(self):
        if self is DetectorType.YOLOv3:
            return YOLOv3
        elif self is DetectorType.Caffe_SSDMobileNet:
            return Caffe_SSDMobileNet
        elif self is DetectorType.TF_SSDMobileNetV2:
            return TF_SSDMobileNetV2
        else:
            raise ValueError(f'Invalid detector type: {self}')
        