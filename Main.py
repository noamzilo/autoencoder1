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


if __name__ == "__main__":
    def main():
        """
         This is just a mockup of the notebook report
        """

        # imports
        from Utils.ConfigProvider import ConfigProvider
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from IPython.display import display
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        config = ConfigProvider.config()
        plt.close('all')

        from noise_cleaning.NoiseCleaner import NoiseCleaner
        noise_cleaner = NoiseCleaner()

        # read data
        inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')

        inspected = noise_cleaner.clean_salt_and_pepper(inspected, 5)
        reference = noise_cleaner.clean_salt_and_pepper(reference, 5)



        # alignment
        from alignment.Aligner import Aligner
        aligner = Aligner()
        # tform = aligner.align_using_ecc(inspected, reference)
        # tform = np.hstack([tform, np.array([0, 0, 1])])
        # print(f"tform: {tform}")

        # aligner.align_using_tform(reference, inspected, tform)

        resize = 5
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(cv2.resize(reference,
                                                                               (0, 0),
                                                                               fx=resize,
                                                                               fy=resize),
                                                                    cv2.resize(inspected, (0, 0), fx=resize, fy=resize))
        moving_should_be_strided_by = np.array(moving_should_be_strided_by_10) / resize

        warped, warp_mask = aligner.align_using_shift(reference, inspected, moving_should_be_strided_by)
        plot_image(warped, "warped")
        # plt.show()

        diff = np.zeros(reference.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(reference))))[warp_mask]
        # diff[~warp_mask] = 0
        # also get rid of registration inaccuracy on the frame
        diff[noise_cleaner.dilate((~warp_mask).astype('uint8'), 2) > 0] = 0

        plot_image(diff.astype('uint8'), "diff")
        # plt.show()

        show_color_diff(warped, reference, "color diff")
        # plt.show()

        diff_blured = noise_cleaner.blur(diff, sigma=5)
        plot_image(diff_blured, "diff_blured")
        # plt.show()

        high_defect_thres = 30
        high_defect_mask_bad = high_defect_thres < diff_blured
        plot_image(high_defect_mask_bad, "high_defect_mask_bad")
        # plt.show()
        # this still leaves edges in as defects

        edges = cv2.Canny(reference.astype('uint8'), 100, 200) > 0
        edges_dialated = noise_cleaner.dilate(edges.astype(np.float32), 3)
        diff_no_edges = diff.copy()
        diff_no_edges[edges_dialated > 0] = 0

        plot_image(edges, "edges")
        plot_image(edges_dialated, "edges_dilated")
        plot_image(diff_no_edges, "diff_no_edges")
        # plt.show()

        high_defect_mask = diff_no_edges > 30
        plot_image(high_defect_mask, "high_defect_mask")

        plt.show()

        hi=5
    main()
