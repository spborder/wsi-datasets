"""

Testing functionality of WSIClassificationDataset

"""

import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt
from classification_dataset import WSIClassificationDataset

from base import WSI


def main():
    slide_path = "./example_data/histology_image.svs"
    ann_path = "./example_data/histology_annotations.geojson"

    test_slide = WSI(
        slide_filepath=slide_path,
        annotations=ann_path,
        attr_kwargs={"test_label": "blah"},
    )
    print(f"{len(test_slide)=}")

    wsi_dataset = WSIClassificationDataset(
        slides=[test_slide], classification_attribute="test_label"
    )

    print(wsi_dataset)
    print(len(wsi_dataset))

    new_tile, new_label = wsi_dataset[0]

    fig, ax = plt.subplots()
    plt.imshow(new_tile)
    plt.title(new_label)
    plt.savefig("./example_data/slide_classification_patch.png")
    plt.close()


if __name__ == "__main__":
    main()
