from torch.utils.data import DataLoader
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default
import numpy as np


class DataloaderSingleImage(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def iterate(self):
        for i_batch, sample_batched in enumerate(self):
            pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def main():
        dataset = dataset_single_image_default()
        batch_size = 4
        shuffle = True
        num_workers = 0
        loader = DataloaderSingleImage(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        # loader.iterate()
        for i_batch, sample_batched in enumerate(loader):
            plt.figure()
            plt.title(f"i_batch = {i_batch}")
            for i_sample in range(sample_batched.shape[0]):
                ax = plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i_sample + 1)
                ax.set_title(f"Sample #{i_sample}")
                ax.axis("off")
                plt.imshow(sample_batched[i_sample, :, :])
                plt.pause(0.000)
            plt.show()

    main()