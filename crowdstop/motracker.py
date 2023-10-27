import os
import cv2 as cv
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable
import numpy as np
from PIL import Image
import itertools
import csv
from pathlib import Path

from crowdstop.ml.multiple_object_tracker import MultipleObjectTracker
from crowdstop.models.sompt import SomptScene
from crowdstop.models.enums import DetectorType, TrackerType
from crowdstop.moteval import calculate_motmetrics, metrics_motchallenge_files, compute_motchallenge


app = Typer(
    help='Object detections in input video using YOLOv3 trained on COCO dataset.'
)

# dictionary to map detector types to their appropriate paths
detector_models = {
    DetectorType.YOLOv3: {
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.weights',
        'configfile_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.cfg',
        'labels_path': '../../multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json',
    },
    DetectorType.Caffe_SSDMobileNet: {
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/MobileNetSSD_deploy.caffemodel',
        'configfile_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/MobileNetSSD_deploy.prototxt',
        'labels_path': '../../multi-object-tracker/examples/pretrained_models/caffemodel_weights/ssd_mobilenet_caffe_names.json',
    },
    DetectorType.TF_SSDMobileNetV2: {
        'weights_path': '../../multi-object-tracker/examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
        'configfile_path': '../../multi-object-tracker/examples/pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
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

    assert not (output_gif and show_gif), 'Model can only output to screen or a GIF file, not both'

    # Populate env vars needed for detector configs. In production this will be done in Dockerfile
    detector_config = detector_models.get(detector_type)
    assert detector_config, f'Invalid detector config: {detector_type}'
    for k, v in detector_config.items():
        os.environ[f'{detector_type.value}_{k}'.upper()] = v

    # Init multiple object tracker model and dataset scene to analyze
    model = MultipleObjectTracker(tracker, detector_type)
    scene = SomptScene(dataset_dir, scene_num)

    # Set detector and tracker types for the scene
    scene.set_tracker_and_detector(detector_type, tracker)
    
    # Start lazy eval
    # tracks = model.track(scene, show_output=show_gif or output_gif)
    tracks = model.quadtrack(scene, show_output=show_gif or output_gif)
    if limit != -1:
        # Only evaluate first n images
        tracks = itertools.islice(tracks, limit)

    # Ensure the directory exists
    output_file_path = scene.detect_fp
    output_file_path.touch()
    
    # Write annotations to a det.txt file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as det_file:
        det_writer = csv.writer(det_file, delimiter=',')
        # det_writer.writerow(['frame', 'person_id', 'xmin', 'ymin', 'width', 'height'])

        for image, annotation in tracks:
            for ann in annotation:
                det_writer.writerow([ann.frame, ann.person_id, ann.x, ann.y, ann.width, ann.height])

            if show_gif:
                cv.imshow("image", image)

            # stop the loop early for testing
            if ann.frame == 50:
                break
    
    if output_gif:
        save_as_gif([image for image, _ in tracks], output_gif)

    # evaluate metrics after det.txt has been generated
    gtSource = scene.annotation_fp
    detSource = scene.detect_fp
    metrics = calculate_motmetrics(gtSource, detSource, bottom_left=False)
    print(metrics)

    # read generated det.txt file to track each object's movement at the end of frames
    movement_results = model.track_movement_from_det_txt(detSource)
    summary = summarize_direction_counts(movement_results)
    print(summary)

def summarize_direction_counts(result_array):
    """ 
    counts the number of each direction movement
    compares the first and last time each object id is seen
    """
    direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

    for _, _, direction in result_array:
        direction_counts[direction] += 1

    return direction_counts

def save_as_gif(image_arrays: Iterable[np.ndarray], output_fp: Path):
    
    images = [Image.fromarray(arr) for arr in image_arrays]
    first_frame = images.pop(0)
    first_frame.save(output_fp, format="GIF", append_images=images,
        save_all=True, duration=200, loop=0)


if __name__ == '__main__':
    app()