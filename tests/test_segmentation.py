"""

Testing functionality of WSISegmentationDataset

"""

import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt
from segmentation_dataset import WSISegmentationDataset

from base import WSI


def main():
    slide_path = "./example_data/histology_image.svs"
    ann_path = "./example_data/histology_annotations.geojson"

    test_slide = WSI(
        slide_filepath=slide_path,
        annotations=ann_path,
    )
    print(f"{len(test_slide)=}")

    wsi_dataset = WSISegmentationDataset(
        slides=[test_slide],
    )

    print(wsi_dataset)
    print(len(wsi_dataset))

    new_tile, new_mask = wsi_dataset[0]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(new_tile)
    ax[1].imshow(new_mask)
    plt.savefig("./example_data/slide_segmentation_patch.png")
    plt.close()


if __name__ == "__main__":
    main()
