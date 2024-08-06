"""Tests CCF registration."""

import os
import unittest

import ants
import numpy as np
from aind_ccf_reg.preprocess import Masking
from pathlib import Path

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
        pass


if __name__ == "__main__":
    unittest.main()
