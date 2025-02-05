import matplotlib.pyplot as plt
import numpy as np

from deepslope.data.elevation import TiffElevationModel, ElevationModel
from deepslope.data.filter import Filter


def visualize_before_after(before: np.ndarray, after: np.ndarray, title_before="Before", title_after="After", cmap="viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(before, cmap=cmap, aspect='auto')
    axes[0].set_title(title_before)
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(after, cmap=cmap, aspect='auto')
    axes[1].set_title(title_after)
    plt.colorbar(im2, ax=axes[1])

    plt.show()


def main():
    model: ElevationModel = TiffElevationModel(
        'data/TBDEMCB00223/Chesapeake_Topobathy_DEM_v1_161.TIF')
    before = model.get_tile(512, 512, 1024, 1024)
    # min_h = np.min(before)
    # max_h = np.max(before)
    # before = (before - min_h) / (max_h - min_h)
    filter = Filter(cutoff=0.005)
    freq, after = filter(before)
    visualize_before_after(
        before, after, title_before="Original Terrain", title_after="Modified Terrain")


if __name__ == '__main__':
    main()
