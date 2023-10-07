import os
from typing import Any, Iterable
import cv2
import numpy as np
from tqdm import tqdm

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils.misc import draw_tracks

from crowdstop.models.sompt import ImageAnnotation, SomptScene
from crowdstop.models.enums import TrackerType, DetectorType


class MultipleObjectTracker:

    def __init__(self, tracker_type: TrackerType, detector_type: DetectorType) -> None:
        
        # set the tracker
        if tracker_type is TrackerType.CentroidTracker:
            self._tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
        elif tracker_type is TrackerType.CentroidKF_Tracker:
            self._tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
        elif tracker_type is TrackerType.SORT:
            self._tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
        elif tracker_type is TrackerType.IOUTracker:
            self._tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, 
                max_detection_confidence=0.7, tracker_output_format='mot_challenge')
        else:
            raise ValueError('Unsupported tracker type!')

        detector_configs = self._get_detector_configs(detector_type)
        # construct the model
        self._model = detector_type.detector_class(
            confidence_threshold=0.5,
            nms_threshold=0.2,
            draw_bboxes=True,
            use_gpu=False,
            **detector_configs
        )
    
    @classmethod
    def _get_detector_configs(cls, detector_type: DetectorType) -> dict:
        '''
        Read environment variables to get filepaths of necessary configs
        '''
        args = dict()
        for config_name in ['weights_path', 'configfile_path', 'labels_path']:
            env_key = f'{detector_type.value}_{config_name}'.upper()
            config_value = os.getenv(env_key)
            assert config_value, f'Could not find necessary env var {env_key}'
            
            args[config_name] = config_value
        return args
    
    def track(self, scene: SomptScene, show_output: bool = False) -> Iterable[tuple[np.ndarray, list[ImageAnnotation]]]:
        for image in tqdm(scene.frames, total=len(scene)):

            #image = cv2.resize(image.cv2_image(), (700, 500))
            image = image.cv2_image()

            bboxes, confidences, class_ids = self._model.detect(image)
            tracks = self._tracker.update(bboxes, confidences, class_ids)
            annotations: list[ImageAnnotation] = list()
            for track in tracks:
                frame, id, xmin, ymin, width, height, *_ = track
                annotations.append(ImageAnnotation(
                    frame=frame,
                    person_id=id,
                    x=xmin,
                    y=ymin,
                    width=width,
                    height=height
                ))
                
            updated_image = None
            if show_output:
                updated_image = self._model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
                updated_image: np.ndarray = draw_tracks(updated_image, tracks)
                
            yield updated_image, annotations
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
