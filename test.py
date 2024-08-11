#!/usr/bin/env python3

from argparse import Namespace
from time import monotonic_ns, sleep
from typing import TextIO
import numpy as np
import sys

import cv2
from mediapipe.tasks.python.components.containers.detections import (
    DetectionResult,
)
import mediapipe as mp

from main import make_visualizer, parse_args
from utils import detection_result_from_str, detection_result_to_str


def log_detection_result(
    file: TextIO, timestamp: int, detection_result: DetectionResult
):
    # TODO: Log on a separate thread using a queue for better(?) performance
    file.write(f"{timestamp}:{detection_result_to_str(detection_result)}")


def replay_log(log: TextIO, dim_x: int, dim_y: int, args: Namespace):
    # initialize the visualizer
    visualizer, state = make_visualizer(args)

    # This "test" is verified by manual inspection, so create a window
    cv2.namedWindow("Video", flags=(cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))

    # Create a blank frame to display
    # Image has 3 color channels - see https://ai.google.dev/edge/api/mediapipe/python/mp/ImageFormat
    blank_frame = np.zeros((dim_x, dim_y, 3), dtype=np.uint8)
    state.frame = blank_frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(state.frame))

    init_line = log.readline()
    assert init_line, "Input file was empty"
    init_log_timestamp_ms, detections = init_line.split(":")
    init_log_time_ms = int(init_log_timestamp_ms)
    init_real_time_ms = monotonic_ns() // 1_000_000
    for line in log:
        timestamp, detections = line.split(":")
        log_elapsed_ms = int(timestamp) - init_log_time_ms
        real_elapsed_ms = (monotonic_ns() // 1_000_000) - init_real_time_ms
        remaining_ms = log_elapsed_ms - real_elapsed_ms
        assert remaining_ms > 0, "Time has already passed!"
        assert remaining_ms < 10_000, "Delay of more than 10 seconds should not occur"
        sleep(float(remaining_ms) / 1_000)
        visualizer(
            detection_result_from_str(detections.rstrip()),
            mp_image,
            mp.Timestamp.from_seconds(float(timestamp) / 1_000_000),
        )
        cv2.imshow("Video", state.frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 1st arg as input file, 2 and 3 as webcam dimensions, and pass rest to normal arg parser
    with open(sys.argv[1], "r") as input_file:
        dim_x = int(sys.argv[2])
        dim_y = int(sys.argv[3])
        sys.argv = [sys.argv[0]] + sys.argv[4:]
        args = parse_args()
        replay_log(input_file, dim_x, dim_y, args)
