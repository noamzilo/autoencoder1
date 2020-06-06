from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.misc
from Utils.plotting.plot_utils import show_color_diff
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import plot_image_3d
from alignment.Aligner import Aligner
from noise_cleaning.NoiseCleaner import NoiseCleaner

# TODO can get dark/light on gray by simple thresholding on diff_blured


def detect_on_gray_areas(inspected, noise_cleaner, warp_mask, warped):
    diff = np.zeros(inspected.shape, dtype=np.float32)
    diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]
    # diff[~warp_mask] = 0
    # also get rid of registration inaccuracy on the frame
    frame_radius = 3
    diff[noise_cleaner.dilate((~warp_mask).astype('uint8'), frame_radius) > 0] = 0
    plot_image(diff.astype('uint8'), "diff")
    # plt.show()
    show_color_diff(warped, inspected, "color diff")
    # plt.show()
    diff_blured = noise_cleaner.blur(diff, sigma=7)  # 5 finds a false negative on non the defective image set
    plot_image(diff_blured, "diff_blured")
    # plt.show()
    high_defect_thres_diff_blured = 50
    high_defect_mask_diff_blured = high_defect_thres_diff_blured < diff_blured
    plot_image(high_defect_mask_diff_blured, "high_defect_mask_diff_blured")
    # plt.show()
    # this still leaves edges in as defects
    edges = cv2.Canny(warped.astype('uint8'), 100, 200) > 0
    glowy_radius = 5
    edges_dialated = noise_cleaner.dilate(edges.astype(np.float32), glowy_radius)
    diff_no_edges = diff.copy()
    diff_no_edges_blured = noise_cleaner.blur(diff_no_edges, sigma=5)
    diff_no_edges_blured[edges_dialated > 0] = 0
    plot_image(edges, "edges")
    plot_image(edges_dialated, "edges_dilated")
    plot_image(diff_no_edges_blured, "diff_no_edges")
    # plt.show()
    high_defect_thres_diff_no_edges = 35
    high_defect_mask = high_defect_thres_diff_no_edges < diff_no_edges_blured
    plot_image(high_defect_mask, "high_defect_mask")
    high_defect_mask_closure = noise_cleaner.close(high_defect_mask.astype('uint8'), diameter=20)
    # This will cause false positives if many nearby defects, but this isn't probable in the business domain.
    # TODO This will also cause false positives on thread-like defects.
    plot_image(high_defect_mask_closure, "high_defect_mask_closure")

    total_defect_mask = np.logical_or(high_defect_mask_diff_blured, high_defect_mask_closure)
    plot_image(total_defect_mask, "total_defect_mask")

    return total_defect_mask

if __name__ == "__main__":
    def main():
        """
         This is just a mockup of the notebook report
        """
        from Utils.ConfigProvider import ConfigProvider
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        config = ConfigProvider.config()
        plt.close('all')

        # read data
        inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')
        # inspected = cv2.imread(config.data.defective_inspected_path2, 0).astype('float32')
        # reference = cv2.imread(config.data.defective_reference_path2, 0).astype('float32')
        # inspected = cv2.imread(config.data.non_defective_inspected_path, 0).astype('float32')
        # reference = cv2.imread(config.data.non_defective_reference_path, 0).astype('float32')

        # clean noise
        noise_cleaner = NoiseCleaner()
        inspected_clean = noise_cleaner.clean_salt_and_pepper(inspected, 5)
        reference_clean = noise_cleaner.clean_salt_and_pepper(reference, 5)

        # registration
        aligner = Aligner()
        resize = 5  # subpixel accuracy resolution
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(static=cv2.resize(inspected_clean,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize),
                                                                       moving=cv2.resize(reference_clean,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize))
        moving_should_be_strided_by = np.array(moving_should_be_strided_by_10) / resize

        warped, warp_mask = aligner.align_using_shift(inspected, reference, moving_should_be_strided_by)
        plot_image(warped, "warped")
        # plt.show()

        detect_on_gray_areas(inspected, noise_cleaner, warp_mask, warped)

        plt.show()
    main()
