"""
Plot functions for easy and fast visualiztaion of images and
regsitration results
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_antsimgs(ants_img, figpath, title="", vmin=0, vmax=500):
    """
    Plot ANTs image

    Parameters
    ------------
    ants_img: ANTsImage
    figpath: PathLike
        Path where the plot is going to be saved
    title: str
        Figure title
    vmin: float
        Set the color limits of the current image.
    vmax: float
        Set the color limits of the current image.
    """

    if figpath:
        ants_img = ants_img.numpy()
        half_size = np.array(ants_img.shape) // 2
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        ax[0].imshow(
            ants_img[half_size[0], :, :], cmap="gray", vmin=vmin, vmax=vmax
        )
        ax[1].imshow(
            ants_img[:, half_size[1], :], cmap="gray", vmin=vmin, vmax=vmax
        )
        im = ax[2].imshow(
            ants_img[
                :,
                :,
                half_size[2],
            ],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        fig.suptitle(title, y=0.9)
        plt.colorbar(
            im, ax=ax.ravel().tolist(), fraction=0.1, pad=0.025, shrink=0.7
        )
        plt.savefig(figpath, bbox_inches="tight", pad_inches=0.1)


def plot_reg(
    moving, fixed, warped, figpath, title="", loc=0, vmin=0, vmax=1.5
):
    """
    Plot registration results: moving, fixed, deformed,
    overlay and difference images after registration

    Parameters
    ------------
    moving: ANTsImage
        Moving image
    fixed: ANTsImage
        Fixed image
    warped: ANTsImage
        Deformed image
    figpath: PathLike
        Path where the plot is going to be saved
    title: str
        Figure title
    loc: int
        Visualization direction
    vmin, vmax: float
        Set the color limits of the current image.
    """

    if loc >= len(moving.shape):
        raise ValueError(
            f"loc {loc} is not allowed, should less than {len(moving.shape)}"
        )

    half_size_moving = np.array(moving.shape) // 2
    half_size_fixed = np.array(fixed.shape) // 2
    half_size_warped = np.array(warped.shape) // 2

    if loc == 0:
        moving = moving.view()[half_size_moving[0], :, :]
        fixed = fixed.view()[half_size_fixed[0], :, :]
        warped = warped.view()[half_size_warped[0], :, :]
        y = 0.75
    elif loc == 1:
        # moving = np.rot90(moving.view()[:,half_size[1], :], 3)
        moving = moving.view()[:, half_size_moving[1], :]
        fixed = fixed.view()[:, half_size_fixed[1], :]
        warped = warped.view()[:, half_size_warped[1], :]
        y = 0.82
    elif loc == 2:
        moving = np.rot90(np.fliplr(moving.view()[:, :, half_size_moving[2]]))
        fixed = np.rot90(np.fliplr(fixed.view()[:, :, half_size_fixed[2]]))
        warped = np.rot90(np.fliplr(warped.view()[:, :, half_size_warped[2]]))
        y = 0.82
    else:
        raise ValueError(
            f"loc {loc} is not allowed. Allowed values are: 0, 1, 2"
        )

    # combine deformed and fixed images to an RGB image
    overlay = np.stack((warped, fixed, warped), axis=2)
    diff = fixed - warped

    fontsize = 14

    fig, ax = plt.subplots(1, 5, figsize=(16, 6))
    ax[0].imshow(moving, cmap="gray", vmin=vmin, vmax=vmax)
    ax[1].imshow(fixed, cmap="gray", vmin=vmin, vmax=vmax)
    ax[2].imshow(warped, cmap="gray", vmin=vmin, vmax=vmax)
    ax[3].imshow(overlay)
    ax[4].imshow(diff, cmap="gray", vmin=-(vmax), vmax=vmax)

    ax[0].set_title("Moving", fontsize=fontsize)
    ax[1].set_title("Fixed", fontsize=fontsize)
    ax[2].set_title("Deformed", fontsize=fontsize)
    ax[3].set_title("Deformed Overlay Fixed", fontsize=fontsize)
    ax[4].set_title("Fixed - Deformed", fontsize=fontsize)

    fig.suptitle(title, size=18, y=y)

    if figpath:
        plt.savefig(figpath, bbox_inches="tight", pad_inches=0.01)
        plt.close()
    else:
        fig.show()
