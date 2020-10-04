from torch.utils.data import Dataset, DataLoader
from Utils.ConfigProvider import ConfigProvider
import cv2
import os


class DatasetSingleImage(Dataset):
    def __init__(self, image_path):
        self._path = image_path
        assert os.path.isfile(self._path)
        self._im = cv2.imread(self._path)
        self._shape = self._im.shape

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self._im

if __name__ == "__main__":
    path = ConfigProvider.config().data.defective_inspected_path1
    dataset = DatasetSingleImage(path)
    for i in range(len(dataset)):
        sample = dataset[i]
        cv2.imshow(f"sample #{i}", sample)
        cv2.waitKey(0)