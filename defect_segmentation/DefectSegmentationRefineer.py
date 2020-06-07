from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
import matplotlib.pyplot as plt


class DefectSegmentationRefiner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()

        self._min_diff_threshold = self._config.refinement.min_diff_threshold
        self._dilation_diameter = self._config.refinement.dilation_diameter
        self._min_new_connected_component_size = self._config.refinement.min_new_connected_component_size

    def refine_segmentation(self, focused_defect_mask, inspected, warped, warp_mask):
        """
        see also Segmenter.infer_region_statistics() for more details.
        Following commented is the start of a solution that I didn't have time to complete, but is another idea:
        After a good segmentation, statistics of pixel color per segment type (there are 3) can be obtained.
        Then, each pixel can be classified as defect/good by its color relative to its segment.
        Early research showed there are 3 gaussian peaks, roughly centered at gray levels 50, 100, 150.

        I didn't have time to create a full working solution like this because my segmentation was lacking.

        max_value_per_pixel = np.zeros_like(diff)
        min_value_per_pixel = np.zeros_like(diff)
        for c in range(config.segmentation.num_classes):
            max_value_per_pixel[warped_segmented == c] = statistics_per_class[c][0] + 2 * statistics_per_class[c][1]
            min_value_per_pixel[warped_segmented == c] = statistics_per_class[c][0] - 2 * statistics_per_class[c][1]

        clean_defect_mask = np.zeros_like(dirty_defect_mask)
        clean_defect_mask[dirty_defect_mask] = max_value_per_pixel[dirty_defect_mask] < inspected[dirty_defect_mask]
        clean_defect_mask[dirty_defect_mask] = np.logical_or(clean_defect_mask[dirty_defect_mask], min_value_per_pixel[dirty_defect_mask] < inspected[dirty_defect_mask])
        plot_image(max_value_per_pixel, "max_value_per_pixel")
        plot_image(min_value_per_pixel, "min_value_per_pixel")
        """

        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]

        diff = self._noise_cleaner.clean_frame(diff, warp_mask)

        # enlarge detection area in case of close proximity misses
        dilated_defect_mask = \
            self._noise_cleaner.dilate(focused_defect_mask.copy().astype('uint8'), diameter=self._dilation_diameter).astype(np.bool)

        diff_above_thres_mask = self._min_diff_threshold < diff
        dilated_defect_mask_with_artifacts = np.logical_and(diff_above_thres_mask, dilated_defect_mask)

        # clean stray pixels which were added due to the dilation, and passed the threshold
        enlarged_defect_mask_too_clean = self._noise_cleaner.clean_stray_pixels_bw(
            dilated_defect_mask_with_artifacts,
            min_size=self._min_new_connected_component_size)

        # but not ones that were present before dilation
        clean_defect_mask = np.logical_or(enlarged_defect_mask_too_clean, focused_defect_mask)

        plot_image(inspected, "inspected")
        plot_image(focused_defect_mask, "focused_defect_mask")
        plot_image(clean_defect_mask, "clean_defect_mask")
        return clean_defect_mask
