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

    def quadtrack(self, scene: SomptScene, show_output: bool = False) -> Iterable[tuple[np.ndarray, list[ImageAnnotation]]]:
        for image in tqdm(scene.frames, total=len(scene)):

            #image = cv2.resize(image.cv2_image(), (700, 500))
            image = image.cv2_image()

            #split image into quadrants
            height, width, channels = image.shape
            x_mid, y_mid = width // 2, height // 2

            top_left = image[0:y_mid, 0:x_mid]
            top_right = image[0:y_mid, x_mid:width]
            bottom_left = image[y_mid:height, 0:x_mid]
            bottom_right = image[y_mid:height, x_mid:width]

            bboxes = []
            confidences = []
            class_ids = []
            offsets = [(0, 0), (x_mid, 0), (0, y_mid), (x_mid, y_mid)]

            for i, subimage in enumerate([top_left, top_right, bottom_left, bottom_right]):
                bboxes_sub, confidences_sub, class_ids_sub = self._model.detect(subimage)

                #applies pixel offset depending on which quadrant is being scanned
                offset_x, offset_y = offsets[i]
                bboxes_sub[:, 0] += offset_x
                bboxes_sub[:, 1] += offset_y

                bboxes.append(bboxes_sub)
                confidences.append(confidences_sub)
                class_ids.append(class_ids_sub)

            bboxes = np.vstack(bboxes)
            confidences = np.concatenate(confidences)
            class_ids = np.concatenate(class_ids)

            # filter to only include "person" class, or id == 0
            person_mask = class_ids == 0
            bboxes = bboxes[person_mask]
            confidences = confidences[person_mask]
            class_ids = class_ids[person_mask]

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
        
    def track_movement(self, annotations_by_frame: Iterable[list[ImageAnnotation]]):
        raise NotImplementedError()
        
    
    def track_movement_from_det_txt(self, det_file_path: str):
        """ compares the first and last positions of each id to show their overall movement """
        # store initial and final positions of objects
        initial_positions = {} 
        final_positions = {}
        # store each (id, distance, direction)
        result_array = []

        with open(det_file_path, 'r') as det_file:

            for line in det_file:
                # read from det.txt
                frame, id, xmin, ymin, width, height, *_ = map(float, line.strip().split(','))

                # update initial positions if not already set
                if id not in initial_positions:
                    initial_positions[id] = (xmin, ymin)

                final_positions[id] = (xmin, ymin)

        for id, initial_pos in initial_positions.items():
            final_pos = final_positions[id]
            # calculate eucledian distance 
            displacement = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
            # calculate relative direction
            dx = final_pos[0] - initial_pos[0]
            dy = final_pos[1] - initial_pos[1]

            # choose direction based on delta x or y
            if abs(dx) > abs(dy):
                direction = "left" if dx < 0 else "right"
            else:
                direction = "up" if dy < 0 else "down"

            result_array.append([id, displacement, direction])
            # print(f'id: {id}, initial pos: {initial_pos}, final pos: {final_pos}')
        
        return result_array