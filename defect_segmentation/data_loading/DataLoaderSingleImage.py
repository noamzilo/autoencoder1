from torch.utils.data import Dataset, DataLoader


class DataLoaderSingleImage(Dataset):
    def __init__(self, image):
        self._im = image
        