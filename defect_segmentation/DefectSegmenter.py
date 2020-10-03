from defect_segmentation.BaseSegmenter import BaseSegmenter
from overrides import overrides
from Utils.plotting.plot_utils import plot_image
import matplotlib.pyplot as plt


class AutoencoderSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()

    @overrides
    def segment_defects(self, inspected, warped, warp_mask):
        plot_image(inspected, "inspected")
        plt.show()
        return ~warp_mask
