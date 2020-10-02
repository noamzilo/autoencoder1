from abc import ABC, abstractmethod
from overrides import EnforceOverrides


class BaseSegmenter(EnforceOverrides, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def segment_defects(self, inspected, warped, warp_mask):
        raise NotImplementedError
