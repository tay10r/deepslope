import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_geotiff_subset(tiff_path, window_bounds):
    """
    Visualizes a subset of a GeoTIFF file in 3D.

    :param tiff_path: Path to the GeoTIFF file.
    :param window_bounds: Tuple (row_start, row_end, col_start, col_end) defining the region of interest.
    """
    row_start, row_end, col_start, col_end = window_bounds

    # Open GeoTIFF file
    with rasterio.open(tiff_path) as dataset:
        print(dataset.shape)
        # Read the specific window of data
        window = Window.from_slices((row_start, row_end), (col_start, col_end))
        transform = dataset.window_transform(window)
        window = rasterio.windows.Window.from_slices(
            (row_start, row_end), (col_start, col_end))
        elevation = dataset.read(1, window=window)

        # Get affine transform for the window

    # Create coordinate grids
    height, width = elevation.shape
    x = np.linspace(transform.c, transform.c + width * transform.a, width)
    y = np.linspace(transform.f, transform.f + height * transform.e, height)
    X, Y = np.meshgrid(x, y)

    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, elevation, cmap="terrain",
                    linewidth=0, antialiased=True)

    # Labels and title
    ax.set_title("3D Terrain Visualization (Subset)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Elevation (meters)")

    plt.show()


# Example usage: Load a 500x500 pixel subset
tiff_file = "data/TBDEMCB00225/Chesapeake_Topobathy_DEM_v1_148.TIF"
# Change this to target a different part of the map
# roi = (1000, 1500, 1000, 1500)
roi = (0, 2048, 0, 2048)
plot_geotiff_subset(tiff_file, roi)
