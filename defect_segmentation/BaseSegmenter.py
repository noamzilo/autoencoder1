from abc import ABC, abstractmethod
from overrides import EnforceOverrides


class BaseSegmenter(EnforceOverrides):
    def __init__(self):
        pass

    @abstractmethod
    def segment_defects(self, inspected, warped, warp_mask):
        raise NotImplementedError
