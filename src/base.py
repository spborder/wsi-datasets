"""

Base dataset object for other WSI dataset types

"""

import json
import os
import random
from typing import Callable
from uuid import uuid4

import geopandas as gpd
import large_image
import numpy as np
import pandas as pd
import torch
from typing_extensions import Union


class WSI:
    def __init__(
        self,
        slide_filepath: str,
        annotations: Union[str, dict, list[str], list[dict], gpd.GeoDataFrame],
        tile_kwargs: dict = {},
        attr_kwargs: dict = {},
    ):
        self.filepath = slide_filepath
        self.annotations = annotations
        self.tile_kwargs = tile_kwargs
        self._id = uuid4().hex[:24]

        assert os.path.exists(self.filepath)

        self.tile_source = large_image.open(self.filepath)
        self.tile_iterator = self.tile_source.tileIterator(**self.tile_kwargs)
        self.process_annotations()

        for k, v in attr_kwargs.items():
            setattr(self, k, v)

    def process_annotations(self):
        self.geos = gpd.GeoDataFrame()

        if self.annotations is None:
            return
        elif not isinstance(self.annotations, list):
            self.annotations = [self.annotations]

        for ann in self.annotations:
            if isinstance(ann, str):
                try:
                    # GeoPandas can read GeoPackage, GeoJSON, or Shapefile types
                    new_geos = gpd.read_file(ann)
                except Exception as e:
                    with open(ann, "r") as f:
                        json_anns = json.load(f)
                        f.close()

                    new_geos = gpd.GeoDataFrame.from_features(json_anns[0]["features"])
            elif isinstance(ann, dict):
                new_geos = gpd.GeoDataFrame.from_features(ann["features"])
            elif isinstance(ann, gpd.GeoDataFrame):
                new_geos = ann
            else:
                raise TypeError(f"Unsupported annotations type: {type(ann)}")

            if self.geos is None:
                self.geos = new_geos
            else:
                self.geos = pd.concat([self.geos, new_geos], axis=0, ignore_index=True)

    def explore(self, host: str = "0.0.0.0", port: int = 8000):
        # Interactive visualization of slide and annotations
        # TODO: This doesn't quite work with the running tileserver. There might be
        # some other settings required (tile size, max zoom, etc.)
        return self.geos.explore(
            tiles=f"http://{host}:{port}/{self._id}/tiles/{{z}}/{{x}}/{{y}}.png",
            attr=self.filepath,
        )

    def start_tileserver(self, host="0.0.0.0", port=8000):
        from tileserver import TileServer

        self.tileserver = TileServer(slides=[self], host=host, port=port)
        self.tileserver.start()

    def __len__(self):
        new_iterator = self.tile_source.tileIterator()
        return len(list(new_iterator))

    def __getitem__(self, idx):
        return next(self.tile_iterator)

    def __str__(self):
        return self.filepath


class WSIDataset:
    def __init__(
        self,
        slides: list[WSI] = [],
        patch_size: list[int] = [256, 256],
        tile_kwargs: dict = {},
        image_transforms: list[Callable] = [],
        target_transforms: list[Callable] = [],
    ):
        self.slides = slides
        self.patch_size = patch_size
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        self.total_tiles = sum([len(slide) for slide in self.slides])
        self.tile_idx = 0
        self.slide_idx = 0

    def __len__(self):
        return self.total_tiles

    def __getitem__(self, idx: int) -> tuple:
        if self.tile_idx + 1 < len(self.slides[self.slide_idx]):
            self.tile_idx += 1
        elif self.slide_idx + 1 < len(self.slides):
            self.slide_idx += 1
            self.tile_idx = 0
        else:
            raise StopIteration

        # Placeholder for some type of ground truth information
        target = 0

        # Applying image & target transforms
        image = self.slides[self.slide_idx][self.tile_idx]["tile"]
        for transform in self.image_transforms:
            image = transform(image)

        for transform in self.target_transforms:
            target = transform(target)

        return image, target

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.slides)} slides ({self.total_tiles} tiles)"
