"""Tests CCF registration."""

import os
import unittest

import ants
import numpy as np
from aind_ccf_reg.preprocess import Masking


class CCFRegistrationTest(unittest.TestCase):
    """Tests CCF registration."""

    def test_preprocessing(self):
        """Tests preprocessing functions"""

        from aind_ccf_reg.preprocess import (perc_normalization,
                                             write_and_plot_image)

        # test percentile normalization
        img = np.random.rand(3, 3, 3)
        img_norm = perc_normalization(img)
        self.assertIsNotNone(img_norm)

        # test write_and_plot_image
        out = write_and_plot_image(img)
        self.assertIsNone(out)

        # test masking
        img = np.zeros((100, 100, 100))
        img[30:70, 30:70, 30:70] = 100
        ants_img = ants.from_numpy(img)
        mask = Masking(ants_img)
        mask = mask.run()
        self.assertIsNotNone(mask)

    def test_ccf_registration(self):
        """Tests CCF registration of an image."""
        from aind_ccf_reg.utils import create_folder
        from sklearn.metrics import f1_score

        results_folder = "../results/test_ccf_registration"
        create_folder(results_folder)

        reg_folder = os.path.abspath(f"{results_folder}/registration_metadata")
        create_folder(reg_folder)

        # path to SPIM template, CCF and template-to-CCF registration
        data_dir = "../data/lightsheet_template_ccf_registration/"
        template_path = os.path.abspath(
            f"{data_dir}/smartspim_lca_template_25.nii.gz"
        )
        ccf_reference_path = os.path.abspath(
            f"{data_dir}/ccf_average_template_25.nii.gz"
        )
        template_to_ccf_transform_warp_path = os.path.abspath(
            f"{data_dir}/spim_template_to_ccf_syn_1Warp.nii.gz"
        )
        template_to_ccf_transform_affine_path = os.path.abspath(
            f"{data_dir}/spim_template_to_ccf_syn_0GenericAffine.mat"
        )
        template_to_ccf_transform_path = [
            template_to_ccf_transform_warp_path,
            template_to_ccf_transform_affine_path,
        ]

        # ensure template, ccf, template-to-ccf transforms exist
        assert os.path.isfile(template_path)
        assert os.path.isfile(ccf_reference_path)
        assert os.path.isfile(template_to_ccf_transform_warp_path)
        assert os.path.isfile(template_to_ccf_transform_affine_path)

        testing_data_dir = "../data/testing_data_for_ccf_registration/"
        input_data_path = (
            f"{testing_data_dir}/registration_metadata/prep_percNorm.nii.gz"
        )
        assert os.path.isfile(input_data_path)

        ants_img = ants.image_read(os.path.abspath(input_data_path))  #
        ants_template = ants.image_read(
            os.path.abspath(template_path)
        )  # SPIM template
        print(f"ants_img: {ants_img}")
        print(f"ants_template: {ants_template}")
        print(
            f"template_to_ccf_transform_path: {template_to_ccf_transform_path}"
        )

        registration_params = {
            "fixed": ants_template,
            "moving": ants_img,
            "type_of_transform": "Rigid",
            "outprefix": f"{reg_folder}/ls_to_template_rigid_",
        }
        rigid_reg = ants.registration(**registration_params)
        aligned_image = rigid_reg["warpedmovout"]

        mask_template = ants_template.numpy() > 0.1
        mask_brain = aligned_image.numpy() > 0.1

        f1_value = f1_score(
            mask_template.flatten(), mask_brain.flatten(), zero_division=np.nan
        )
        print(f"** f1_value: {f1_value} **")
        self.assertTrue(f1_value > 0.70)


if __name__ == "__main__":
    unittest.main()
