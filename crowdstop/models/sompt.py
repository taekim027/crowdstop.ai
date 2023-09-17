from collections import defaultdict
import os
from pathlib import Path
from collections.abc import Iterable
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import csv
from pydantic import BaseModel

np.random.seed(0)
COLORS = [
    tuple(np.random.choice(range(256), size=3))
    for _ in range(1000)
]

class ImageAnnotation(BaseModel):
    frame: int
    person_id: int
    x: int
    y: int
    width: int
    height: int

    def as_coordinates(self):
        return self.x, self.y, self.x + self.width, self.y + self.height
    
    def center_top(self):
        return self.x + self.width / 2, self.y + self.height / 2

class SomptImage:

    def __init__(self, scene: 'SomptScene', frame: int) -> None:
        self._scene = scene
        self._frame = frame
        self._fp = self._scene.image_dir / f'{str(frame).zfill(6)}.jpg'
        assert self._fp.exists() and self._fp.is_file(), \
            f'{self._fp} does not exist'

    def image(self) -> Image:
        return Image.open(self._fp)
    
    def render(self):
        img = self.image()
        draw = ImageDraw.Draw(img)
        for anno in self._scene.get_annotations(self._frame):
            draw.rectangle(anno.as_coordinates(), outline=COLORS[anno.person_id], width=2)
            # draw.text(anno.center_top(), str(anno.person_id), fill='red', font=)
        return img
        
class SomptScene:
    def __init__(self, dataset_dir: str, number: int, train: bool = True) -> None:
        self._dir = dataset_dir / ('train' if train else 'test') / f'SOMPT22-{str(number).zfill(2)}'
        assert self._dir.exists() and self._dir.is_dir(), \
            f'Cannot find scene "{number}"'
        
        annotations: dict[str, list[ImageAnnotation]] = defaultdict(list)
        with open(self.annotation_fp, 'r') as gt_f:
            reader = csv.DictReader(gt_f, fieldnames=list(ImageAnnotation.model_fields))
            for row in reader:
                row.pop(None)   # Remove extraneous fields
                annotations[int(row['frame'])].append(ImageAnnotation(**row))
        self._annotations = dict(annotations)   # defaultdict can lead to hard to catch bugs

    @property
    def image_dir(self) -> Path:
        return self._dir / 'img1'
    
    @property
    def annotation_fp(self) -> Path:
        return self._dir / 'gt' / 'gt.txt'

    def __len__(self) -> int:
        return len(list(self.image_dir.glob('*.jpg')))
    
    def get_frame(self, number: int) -> SomptImage:
        assert 0 < number <= len(self), f'Invalid index, must be between 1 and {len(self)}'
        return SomptImage(self, number)
    
    @property
    def frames(self) -> Iterable[SomptImage]:
        for i in range(len(self)):
            yield self.get_frame(i + 1)
    
    def get_annotations(self, number: int) -> list[ImageAnnotation]:
        assert 0 < number <= len(self), f'Invalid index, must be between 1 and {len(self)}'
        return self._annotations[number]
