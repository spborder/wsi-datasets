"""

A subclass of the WSIDataset class specifically for handling segmentation tasks

- Control how target masks are generated from provided annotations
- Implement class balancing based on mask-class content
- Retrieve relative pixel-counts/percentages of classes/number of tiles containing each class

- Some method for combining segmentation results on adjacent slides
- Some method for resolving overlapping classes

"""

import warnings
from typing import Callable

import rasterio.features
import torch
from shapely.geometry import box
from typing_extensions import Union

from base import WSI, WSIDataset


class WSISegmentationDataset(WSIDataset):
    def __init__(
        self,
        slides: list[WSI] = [],
        patch_size: list[int] = [256, 256],
        tile_kwargs: dict = {},
        image_transforms: list[Callable] = [],
        target_transforms: list[Callable] = [],
    ):
        super().__init__(
            slides,
            patch_size,
            tile_kwargs,
            image_transforms,
            target_transforms,
        )

    def __getitem__(self, idx: int) -> tuple:
        if self.tile_idx + 1 < len(self.slides[self.slide_idx]):
            self.tile_idx += 1
        elif self.slide_idx + 1 < len(self.slides):
            self.slide_idx += 1
            self.tile_idx = 0
        else:
            raise StopIteration

        next_tile_object = self.slides[self.slide_idx][self.tile_idx]
        tile_bbox = [
            next_tile_object["x"],
            next_tile_object["y"],
            next_tile_object["x"] + next_tile_object["width"],
            next_tile_object["y"] + next_tile_object["height"],
        ]

        # Need something to indicate different layers and indices
        intersecting_annotations = self.slides[self.slide_idx].geos.intersection(
            box(*tile_bbox)
        )

        with warnings.catch_warnings(action="ignore"):
            # Apply the class index here
            mask = rasterio.features.rasterize(
                shapes=intersecting_annotations.geometry.tolist(),
                out_shape=(next_tile_object["height"], next_tile_object["width"]),
            )

        return next_tile_object["tile"], mask
