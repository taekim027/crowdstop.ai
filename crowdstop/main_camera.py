import os
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable
from datetime import datetime
import requests
from pathlib import Path
import logging
import random

from crowdstop.ml.multiple_object_tracker import MultipleObjectTracker
from crowdstop.models.sompt import SomptScene
from crowdstop.models.enums import DetectorType, TrackerType
from crowdstop.models.api import CameraCreateRequest, CameraUpdateRequest, PlaceCreateRequest, Velocity

logger = logging.getLogger(__file__)

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
    host_url: str,
    dataset_dir: Annotated[Path, Argument(help='Base dataset directory containing SOMPT scenes')],
    scene_num: int,
    detector_type: Annotated[DetectorType, Option('--detector_type', '-d', help='Detector used to detect objects')] = DetectorType.YOLOv3,
    tracker: Annotated[TrackerType, Option('--tracker', '-t', help='Tracker used to track objects')] = TrackerType.IOUTracker,
    update_frequency: Annotated[int, Option('--update_frequency', '-f', help='Frequency of updates to server in number of frames')] = 25
) -> None:
    
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
    
    place_ids = []
    for place in places:
        r = requests.post(
            url=f'{host_url}/place',
            json=PlaceCreateRequest(latitude=random.randint(0, 100), longitude=random.randint(0, 100), area=10).model_dump()
        )
        r.raise_for_status()
        place_ids.append(r.json()['uuid'])


    response = requests.post(
        url=f'{host_url}/camera',
        json=CameraCreateRequest(latitude=10, longitude=10, area=10, place_ids=place_ids).model_dump()
    )
    response.raise_for_status()
    camera_id = response.json()['uuid']
    logger.info(f'Received camera id {camera_id}')
    
    # Start lazy eval
    tracks = model.quadtrack(scene, show_output=False)
    
    for i, (_, annotations) in enumerate(tracks):
        # TODO: track cumulative number of people who went from one side to another
        
        if i % update_frequency == 0:
            
            request = CameraUpdateRequest(
                timestamp=str(datetime.now()),
                count=len(annotations),
                velocities=[]       # TODO: Include velocities
            )
            
            r = requests.put(
                url=f'{host_url}/camera/{id}',
                json=request.model_dump()
            )
            r.raise_for_status()
        

def summarize_direction_counts(result_array):
    """ 
    counts the number of each direction movement
    compares the first and last time each object id is seen
    """
    direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

    for _, _, direction in result_array:
        direction_counts[direction] += 1

    return direction_counts


if __name__ == '__main__':
    app()