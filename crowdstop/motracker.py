import cv2 as cv
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable
from enum import Enum
import numpy as np
from PIL import Image
from tqdm import tqdm

from motrackers.detectors import YOLOv3
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

@app.command()
def main(
    dataset_dir: Annotated[Path, Argument(help='Base dataset directory containing SOMPT scenes')],
    scene_num: int,
    weights: Annotated[Path, Option('--weights', '-w', help='Path to weights file of YOLOv3 (`.weights` file)')] = '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.weights',
    config: Annotated[Path, Option('--config', '-c', help='Path to config file of YOLOv3 (`.cfg` file)')] = '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.cfg',
    labels: Annotated[Path, Option('--labels', '-l', help='Path to labels file of coco dataset (`.names` file')] = '../../multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json',
    tracker: Annotated[TrackerType, Option('--tracker', '-t', help='Tracker used to track objects')] = TrackerType.CentroidKF_Tracker.value,
    gpu: Annotated[bool, Option(help='Flag to use gpu to run the deep learning model. Default is `False`')] = False,
    output_gif: Path = None,
    show_gif: bool = True,
    limit: int = -1
) -> None:

    if tracker is TrackerType.CentroidTracker:
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif tracker is TrackerType.CentroidKF_Tracker:
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif tracker is TrackerType.SORT:
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif tracker is TrackerType.IOUTracker:
        tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')

    model = YOLOv3(
        weights_path=str(weights),
        configfile_path=str(config),
        labels_path=str(labels),
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