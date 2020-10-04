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
        for i_batch, sample_batched in enumerate(loader):
            fig, axs = plt.subplots(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)))
            fig.suptitle(f"i_batch = {i_batch}")
            for i_sample, ax in zip(range(sample_batched.shape[0]), axs.flat):
                # ax = plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i_sample + 1)
                ax.set_title(f"Sample #{i_sample}")
                ax.axis("off")
                ax.imshow(sample_batched[i_sample, :, :])
                plt.pause(0.001)



    main()
