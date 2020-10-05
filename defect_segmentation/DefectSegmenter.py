from defect_segmentation.BaseSegmenter import BaseSegmenter
from overrides import overrides
from Utils.plotting.plot_utils import plot_image
import matplotlib.pyplot as plt
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default


class AutoencoderSegmenter(BaseSegmenter):
    def __init__(self):
        super().__init__()
        dataset = dataset_single_image_default()

    @overrides
    def segment_defects(self, inspected, warped, warp_mask):
        plot_image(inspected, "inspected")
        plt.show()
        return ~warp_mask
