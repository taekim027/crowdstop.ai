import os
import cv2 as cv
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable, NamedTuple
import numpy as np
from PIL import Image
import itertools
import csv
import json
import time
from pathlib import Path
from shapely.geometry.polygon import Polygon

from crowdstop.ml.multiple_object_tracker import MultipleObjectTracker
from crowdstop.models.sompt import SomptScene
from crowdstop.models.enums import DetectorType, TrackerType
from scripts.moteval import calculate_motmetrics, metrics_motchallenge_files, compute_motchallenge


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

# zone config path
zone_config_path = '../ml/zone_config.json'

# Define a named tuple for annotations
Annotation = NamedTuple('Annotation', [('frame', int), ('person_id', int), ('x', int), ('y', int), ('width', int), ('height', int)])


@app.command()
def main(
    dataset_dir: Annotated[Path, Argument(help='Base dataset directory containing SOMPT scenes')],
    scene_num: int,
    detector_type: Annotated[DetectorType, Option('--detector_type', '-d', help='Detector used to detect objects')],
    tracker: Annotated[TrackerType, Option('--tracker', '-t', help='Tracker used to track objects')],
    gpu: Annotated[bool, Option(help='Flag to use gpu to run the deep learning model. Default is `False`')] = False,
    output_gif: Path = None,
    show_gif: bool = True,
    limit: int = 50
) -> None:
    
    print(os.getcwd())
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
    downsample_rate = 5
    # tracks = model.track(scene, show_output=show_gif or output_gif, downsample_rate=downsample_rate, limit=limit)
    tracks = model.quadtrack(scene, show_output=show_gif or output_gif, downsample_rate = downsample_rate, limit=limit)

    # # Sample a subset of frames
    # mod_val = 20
    # # Handle for when limit == -1
    # if limit != -1:
    #     sampled_tracks = itertools.islice(tracks, 0, limit, mod_val)
    # else:
    #     sampled_tracks = itertools.islice(tracks, 0, None, mod_val)

    # Ensure the directory exists
    output_file_path = scene.detect_fp
    output_file_path.touch()
    
    # Write annotations to a det.txt file
    start_time = time.time()
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as det_file:
        det_writer = csv.writer(det_file, delimiter=',')
        # det_writer.writerow(['frame', 'person_id', 'xmin', 'ymin', 'width', 'height'])

        for image, annotation in tracks:

            for ann in annotation:
                det_writer.writerow([ann.frame, ann.person_id, ann.x, ann.y, ann.width, ann.height])

            if show_gif:
                cv.imshow("image", image)
    
    duration = time.time() - start_time
    print(f'duration: {duration}')

    if output_gif:
        save_as_gif([image for image, _ in tracks], output_gif)

    # evaluate metrics after det.txt has been generated
    gtSource = scene.annotation_fp
    detSource = scene.detect_fp
    metrics = calculate_motmetrics(gtSource, detSource, bottom_left=False, sample_rate=downsample_rate)
    print(metrics)

    # read generated det.txt file to track each object's movement at the end of frames
    #movement_results = model.track_movement_from_det_txt(detSource)

    # read zone configs
    # with open(zone_config_path, 'r') as config_file:
    #     zone_configs = json.load(config_file)
    
    # zones = [Polygon(zone) for zone in zone_configs[str(scene_num)]]

    # # read det.txt file to track zone movements
    # movement_results = model.track_zone_movement(detSource, zones)
    # #summary = summarize_direction_counts(movement_results)
    # #print(summary)
    # print(movement_results)


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