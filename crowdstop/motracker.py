import cv2 as cv
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable
from enum import Enum
import numpy as np
from PIL import Image
from tqdm import tqdm

from motrackers.detectors import YOLOv3, Caffe_SSDMobileNet, TF_SSDMobileNetV2
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

from crowdstop.models.sompt import SomptScene


app = Typer(
    help='Object detections in input video using YOLOv3 trained on COCO dataset.'
)


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

# dictionary to map detector types to their appropriate paths
detector_models = {
    DetectorType.YOLOv3: {
        'model_class': YOLOv3,
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.weights',
        'config_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.cfg',
        'labels_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json',
    },
    DetectorType.Caffe_SSDMobileNet: {
        'model_class': Caffe_SSDMobileNet,
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/MobileNetSSD_deploy.caffemodel',
        'config_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/MobileNetSSD_deploy.prototxt',
        'labels_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/ssd_mobilenet_caffe_names.json',
    },
    DetectorType.TF_SSDMobileNetV2: {
        'model_class': TF_SSDMobileNetV2,
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
        'config_path': '../../multi-object-tracker/examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
        'labels_path': '../../multi-object-tracker/examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_names.json',
    },
}

@app.command()
def main(
    dataset_dir: Annotated[Path, Argument(help='Base dataset directory containing SOMPT scenes')],
    scene_num: int,
    detector_type: Annotated[DetectorType, Option('--detector_type', '-d', help='Detector used to detect objects')],
    tracker: Annotated[TrackerType, Option('--tracker', '-t', help='Tracker used to track objects')],
    gpu: Annotated[bool, Option(help='Flag to use gpu to run the deep learning model. Default is `False`')] = False,
    output_gif: Path = None,
    show_gif: bool = True,
    limit: int = -1
) -> None:

    # set the tracker
    if tracker is TrackerType.CentroidTracker:
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif tracker is TrackerType.CentroidKF_Tracker:
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif tracker is TrackerType.SORT:
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif tracker is TrackerType.IOUTracker:
        tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')

    # get the selected detector model and relevant paths
    detector = detector_models.get(detector_type)
    if detector is None:
        raise ValueError("Unsupported detector type")

    model_class = detector['model_class']
    weights_path = detector['weights_path']
    config_path = detector['config_path']
    labels_path = detector['labels_path']

    # construct the model
    model = model_class(
        weights_path=weights_path,
        configfile_path=config_path,
        labels_path=labels_path,
        confidence_threshold=0.5,
        nms_threshold=0.2,
        draw_bboxes=True,
        use_gpu=gpu
    )

    scene = SomptScene(dataset_dir, scene_num)
    images = list(track(scene, model, tracker, show_gif))
    if limit != -1:
        images = images[:limit]
    
    if output_gif:
        save_as_gif(images, output_gif)


def track(scene: SomptScene, model, tracker, show_gif: bool) -> Iterable[np.ndarray]:

    for image in tqdm(scene.frames, total=len(scene)):

        image = cv.resize(image.cv2_image(), (700, 500))

        bboxes, confidences, class_ids = model.detect(image)
        tracks = tracker.update(bboxes, confidences, class_ids)
        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)

        updated_image: np.ndarray = draw_tracks(updated_image, tracks)

        if show_gif:
            cv.imshow("image", updated_image)
        yield updated_image
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()

def save_as_gif(image_arrays: Iterable[np.ndarray], output_fp: Path):
    
    images = [Image.fromarray(arr) for arr in image_arrays]
    first_frame = images.pop(0)
    first_frame.save(output_fp, format="GIF", append_images=images,
        save_all=True, duration=200, loop=0)


if __name__ == '__main__':
    app()