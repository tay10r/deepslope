import numpy as np


class Filter:
    """
    Used for removing frequency components of a terrain.
    """

    def __init__(self, cutoff: float, high_pass: bool = False, normalize_output: bool = False):
        """
        Initializes the filter.

        :cutoff: The cut-off distance from the center of the shifted FFT. Should be between 0 and 1, and usually closer to zero.
        :high_pass: If true, the filter preserves high-pass components.
        """
        self.cutoff = cutoff
        self.high_pass = high_pass
        self.normalize_output = normalize_output

    def __call__(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        freq = np.fft.fft2(data)
        shifted_freq = np.fft.fftshift(freq)
        x = np.linspace(-1, 1, data.shape[0])
        y = np.linspace(-1, 1, data.shape[1])
        x, y = np.meshgrid(x, y)
        mask = np.exp(-(x**2 + y**2) / (self.cutoff**2.0))
        if self.high_pass:
            mask = 1.0 - mask
        result = np.fft.ifft2(np.fft.ifftshift(shifted_freq * mask)).real
        if self.normalize_output:
            min_h = np.min(result)
            max_h = np.max(result)
            result = (result - min_h) / (max_h - min_h)
        return np.log1p(np.abs(shifted_freq)), result
