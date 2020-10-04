from torch.utils.data import Dataset
from Utils.ConfigProvider import ConfigProvider
import cv2
import os
from overrides import overrides


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
            [(c, min(c + self._sample_rows, self._rows)) for c in range(0, self._rows - self._sample_rows, self._stride_rows)]
        self._cols_tuples_range = \
            [(r, min(r + self._sample_cols, self._cols)) for r in range(0, self._cols - self._sample_cols, self._stride_cols)]

        self._n_strides_rows = len(self._rows_tuples_range)
        self._n_strides_cols = len(self._cols_tuples_range)
        self._total_strides = self._n_strides_rows * self._n_strides_cols

    def __len__(self):
        return self._total_strides

    @overrides
    def __getitem__(self, ind):
        row_ind = ind // self._n_strides_cols
        col_ind = ind % self._n_strides_cols
        sample_x = self._rows_tuples_range[row_ind]
        sample_y = self._cols_tuples_range[col_ind]
        sample = self._im[sample_x[0]:sample_x[1], sample_y[0]:sample_y[1]]
        assert sample.shape[:2] == self._sample_shape
        return sample


def dataset_single_image_default():
    path = ConfigProvider.config().data.defective_inspected_path1
    sample_shape = (50, 50)
    strides = (25, 25)
    dataset = DatasetSingleImage(path, sample_shape, strides)
    return dataset


if __name__ == "__main__":
    def main():
        dataset = dataset_single_image_default()
        for i in range(len(dataset)):
            sample = dataset[i]
            # cv2.imshow(f"sample #{i}", sample)
            cv2.imshow(f"sample", sample)
            cv2.waitKey(50)

    main()