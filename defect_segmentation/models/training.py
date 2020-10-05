import torch
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default
import numpy as np


def train_autoencoder():
    seed = 42
    np.random.seed(seed)
    dataset = dataset_single_image_default()

    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = \
        torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed))
    pass


if __name__ == "__main__":
    train_autoencoder()
