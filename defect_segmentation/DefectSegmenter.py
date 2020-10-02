from defect_segmentation.BaseSegmenter import BaseSegmenter
from overrides import overrides


class AutoencoderSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()

    @overrides
    def segment_defects(self, inspected, warped, warp_mask):
        pass
