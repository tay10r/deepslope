"""
This program takes a before and after image and produces a series
of linear interpolated frames between them. It is meant to create
an animation of the effect, to show case what the network is doing.
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from deepslope.config import Config, get_config


def main():
    config: Config = get_config()
    before_path = config.test_image
    after_path = str(list(Path(config.tmp_path).glob('test_*.png'))[-1])

    before = cv2.imread(before_path)
    if before is None:
        logger.error(f'Failed to open "{before_path}"')
        return

    after = cv2.imread(after_path)
    if after is None:
        logger.error(f'Failed to open "{after_path}"')
        return

    height, width, channels = before.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24  # frames per second
    video_filename = str(Path(config.tmp_path) / 'sequence.avi')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    num_frames = fps * 4

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = cv2.addWeighted(before, 1 - alpha, after, alpha, 0)
        out.write(frame)
    out.release()


main()
