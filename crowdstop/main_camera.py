import json
import os
from typer import Typer, Argument, Option
from pathlib import Path
from typing import Annotated, Iterable
from datetime import datetime, timezone
import requests
from pathlib import Path
import logging
from shapely.geometry.polygon import Polygon

from crowdstop.ml.multiple_object_tracker import MultipleObjectTracker
from crowdstop.models.camera_config import CameraConfig
from crowdstop.models.sompt import SomptScene
from crowdstop.models.enums import DetectorType, TrackerType
from crowdstop.models.api import CameraCreateRequest, CameraUpdateRequest, PlaceCreateRequest, Velocity

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
    camera_config_path: Path,
    host_url: str = 'http://localhost:8000',
    detector_type: Annotated[DetectorType, Option('--detector_type', '-d', help='Detector used to detect objects')] = DetectorType.YOLOv3.value,
    tracker: Annotated[TrackerType, Option('--tracker', '-t', help='Tracker used to track objects')] = TrackerType.IOUTracker.value,
    update_frequency: Annotated[int, Option('--update_frequency', '-f', help='Frequency of updates to server in number of frames')] = 25
) -> None:
    
    # Ping server to confirm correct host url
    r = requests.get(
        url=f'{host_url}/health'
    )
    r.raise_for_status()
    print(f'Connected to {host_url}')
    
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
    
    with open(camera_config_path, 'r') as f:
        camera_config = CameraConfig(**json.load(f))
    
    place_ids: list[str] = []
    for place in camera_config.places:
        r = requests.post(
            url=f'{host_url}/place',
            json=PlaceCreateRequest(
                latitude=place.latitude, 
                longitude=place.longitude, 
                area=place.area
            ).model_dump()
        )
        r.raise_for_status()
        place_ids.append(r.json()['uuid'])

    response = requests.post(
        url=f'{host_url}/camera',
        json=CameraCreateRequest(
            latitude=camera_config.latitude, 
            longitude=camera_config.longitude,
            area=camera_config.area, 
            place_ids=place_ids
        ).model_dump()
    )
    response.raise_for_status()
    camera_id = response.json()['uuid']
    print(f'Received camera id {camera_id}')
    
    # Create polygons
    zone_polygons = [
        Polygon(p.polygon)
        for p in camera_config.places
    ]

    # Start lazy eval
    tracks = model.track(scene, show_output=False)
    
    cum_frames = list()
    for i, (_, annotations) in enumerate(tracks):
        cum_frames.append(annotations)
        if i % update_frequency == 0:
            
            # Track movement of people across zones so far
            movement = model.track_movement(zone_polygons, cum_frames)
            velocities = {
                id: m
                for id, m in zip(place_ids, movement)
            }

            request = CameraUpdateRequest(
                timestamp=str(datetime.now().replace(tzinfo=timezone.utc)), 
                count=len(annotations),
                velocities=velocities
            )
            
            r = requests.put(
                url=f'{host_url}/camera/{camera_id}',
                json=request.model_dump()
            )
            r.raise_for_status()

            cum_frames = list()     # reset cumulative frames to not double-count movement
        

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