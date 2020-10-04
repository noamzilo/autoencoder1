from torch.utils.data import Dataset, DataLoader
from Utils.ConfigProvider import ConfigProvider
import cv2
import os


class DatasetSingleImage(Dataset):
    def __init__(self, image_path: str, sample_shape: tuple, strides: tuple):
        self._path = image_path
        assert os.path.isfile(self._path)
        self._im = cv2.imread(self._path)
        self._shape = self._im.shape
        self._rows, self._cols = self._shape[0], self._shape[1]

        self._sample_shape = sample_shape
        self._sample_rows, self._sample_cols = self._sample_shape[0], self._sample_shape[1]

        self._strides = strides
        self._stride_rows, self._stride_cols = self._strides[0], self._strides[1]
        # self._rows_start_range = range(0, self._rows, self._stride_rows)
        # self._cols_start_range = range(0, self._cols, self._stride_cols)

        self._rows_tuples_range = \
            [(c, min(c + self._stride_rows, self._rows)) for c in range(0, self._rows, self._stride_rows)]
        self._cols_tuples_range = \
            [(r, min(r + self._stride_cols, self._cols)) for r in range(0, self._cols, self._stride_cols)]

    def __len__(self):
        return len(self._rows_tuples_range) * len(self._cols_tuples_range)

    def __getitem__(self, ind):
        row_ind = ind // self._cols
        col_ind = ind % self._cols
        sum_im_x = self._rows_tuples_range[row_ind]
        sum_im_y = self._cols_tuples_range[col_ind]
        return self._im[sum_im_x[0]:sum_im_x[1], sum_im_y[0]:sum_im_y[1]]


if __name__ == "__main__":
    def main():
        path = ConfigProvider.config().data.defective_inspected_path1
        sample_shape = (15, 15)
        strides = (1, 1)
        dataset = DatasetSingleImage(path, sample_shape, strides)
        for i in range(len(dataset)):
            sample = dataset[i]
            cv2.imshow(f"sample #{i}", sample)
            cv2.waitKey(0)

    main()