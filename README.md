# WSI Datasets

This repository contains some implementations of Dataset classes for Whole Slide Image (WSI) data for various tasks.

## Datasets

The primary methods that have to be implemented for PyTorch Datasets (`torch.utils.data.Dataset`) are *\__init__*, *\__len__*, and *\__getitem__*. All together, this turns this object into something which can produce an interable when passed to *iter()*

This allows custom objects to be provided to the PyTorch DataLoader class (`torch.utils.data.DataLoader`) which handles preparation of batches of data within training and testing loops.

## Usage

```python
from wsi_datasets.base import WSI
from wsi_datasets.classification_dataloader import WSIClassificationDataset

from torch.utils.data import DataLoader

slide_path = '/path/to/wsi.tiff'
wsi = WSI(slide_filepath = slide_path, attr_kwargs = {'test_label': 'label'})

classification_dataset = WSIClassificationDataset(
    slides = [wsi], 
    classification_attribute='test_label'
)

train_dataloader = DataLoader(classification_dataset, batch_size = 32, shuffle = True)

```
