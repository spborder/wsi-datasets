"""

Testing functionality of base WSIDataset

"""

import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt

from base import WSI, WSIDataset


def main():
    slide_path = "./example_data/histology_image.svs"
    ann_path = "./example_data/histology_annotations.geojson"

    test_slide = WSI(slide_filepath=slide_path, annotations=ann_path)
    print(f"{len(test_slide)=}")

    wsi_dataset = WSIDataLoader(slides=[test_slide])

    print(wsi_dataset)
    print(len(wsi_dataset))

    fig, ax = plt.subplots()
    plt.imshow(wsi_dataset[0][0])
    plt.savefig("./example_data/slide_patch.png")
    plt.close()


if __name__ == "__main__":
    main()
