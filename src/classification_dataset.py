"""

A subclass of the WSIDataset class specifically for handling classification tasks

- Specify classification attribute
- Implement class balancing on the side of the Dataset
- Retrieve class counts

- Classification by Tile or by WSI? Specify at what level the labels should
be applied
- Some implementation of multi-class labeling

"""

from typing import Callable

import torch
from typing_extensions import Union

from base import WSI, WSIDataset


class DynamicTileLabelWSI(WSI):
    def __init__(self, classification_attributes: Union[str, list[str]], **kwargs):
        super().__init__(**kwargs)

        self.classification_attributes = classification_attributes

    def __getitem__(self, idx: int) -> tuple:
        tile_object = super().__getitem__(idx)
        tile_attrs = self._get_tile_attr(tile_object)

        return tile_object, tile_attrs

    def _get_tile_attr(self, tile_object):
        raise NotImplementedError


class WSIClassificationDataset(WSIDataset):
    def __init__(
        self,
        slides: list[WSI] = [],
        classification_attribute: Union[str, list[str]] = [],
        patch_size: list[int] = [256, 256],
        tile_kwargs: dict = {},
        image_transforms: list[Callable] = [],
        target_transforms: list[Callable] = [],
    ):

        self.classification_attribute = classification_attribute

        super().__init__(
            slides,
            patch_size,
            tile_kwargs,
            image_transforms,
            target_transforms,
        )

    def __getitem__(self, idx: int) -> tuple:
        tile, target = super().__getitem__(idx)

        if not isinstance(self.slides[self.slide_idx], DynamicTileLabelWSI):
            if isinstance(self.classification_attribute, str):
                classification_label = getattr(
                    self.slides[self.slide_idx], self.classification_attribute
                )
            elif isinstance(self.classification_attribute, list):
                classification_label = [
                    getattr(self.slides[self.slide_idx], c)
                    for c in self.classification_attribute
                ]
            else:
                classification_label = target
        else:
            # If this is a DynamicTileLabelWSI, then the label is provided along with the tile (tuple)
            tile, classification_label = tile

        return tile, classification_label
