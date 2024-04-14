#!/usr/bin/env python3

import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, Union
import math
import time

# Constants related to image processing
SMOOTH_NUM_FRAMES = 10  # Number of frames to average over
# Normalization factor
SMOOTH_TRIANGLE = SMOOTH_NUM_FRAMES * (SMOOTH_NUM_FRAMES + 1) / 2
# Triangular weighted moving average - weights n, n-1, ... 2, 1
SMOOTHING_WEIGHTS = (
    np.tile(np.arange(SMOOTH_NUM_FRAMES, 0, -1), 4)
    .reshape(4, SMOOTH_NUM_FRAMES)
    .transpose()
)
# Normalize weights
SMOOTHING_WEIGHTS_NORMALIZED = SMOOTHING_WEIGHTS / SMOOTH_TRIANGLE
assert np.array_equal(
    np.sum(SMOOTHING_WEIGHTS_NORMALIZED, axis=0), [1.0] * 4
), "Failed to normalize weights"
LOW_PASS_THRESHOLD = 20  # Changes smaller than this (in pixels) are ignored
INITIAL_SIZE = 400

# Gaps between the edge of the face and the edge of the window in pixels
MARGIN_TOP = 75
MARGIN_BOTTOM = 25
MARGIN_LEFT = 50
MARGIN_RIGHT = 50

# Constants for drawing debug info on screen
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # blue - the format is BGR
DEBUG_LEVEL = 0


# Debug utilities
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class State:
    def __init__(self) -> None:
        self.frame = None


def make_visualizer():
    # Mutable state captured inside visualize
    shared_state = State()
    # Start with a fixed initial size
    trailing_dimensions = np.array([[0, INITIAL_SIZE] * 2] * SMOOTH_NUM_FRAMES)
    low_pass_last_amounts = np.array([0, INITIAL_SIZE] * 2)

    def visualize(detection_result, image, timestamp) -> None:
        """
        Crops the image to the bounding box, drawing bounding boxes and keypoints
        on the input image based on the DEBUG_LEVEL
        """
        opencv_image = np.array(image.numpy_view())
        height, width, _ = opencv_image.shape

        # Choose detecton with the highest probability
        max_probability = 0
        max_detection = None
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.score > max_probability:
                max_probability = category.score
                max_detection = detection

        # If no face was detected, use the new frame with the previous dimensions
        nonlocal shared_state
        nonlocal trailing_dimensions
        if not max_detection:
            shared_state.frame = opencv_image[
                trailing_dimensions[0][0] : trailing_dimensions[0][1],
                trailing_dimensions[0][2] : trailing_dimensions[0][3],
            ]
            return

        bbox = max_detection.bounding_box
        start_x = max(bbox.origin_x - MARGIN_LEFT, 0)
        end_x = min(bbox.origin_x + bbox.width + MARGIN_RIGHT, width)
        start_y = max(bbox.origin_y - MARGIN_TOP, 0)
        end_y = min(bbox.origin_y + bbox.height + MARGIN_BOTTOM, height)
        current_box = np.array([start_y, end_y, start_x, end_x])

        # Ignore changes below a threshold
        nonlocal low_pass_last_amounts
        if np.max(np.abs(current_box - low_pass_last_amounts)) >= LOW_PASS_THRESHOLD:
            low_pass_last_amounts = current_box
        dimensions = low_pass_last_amounts

        # Update the trailing values
        trailing_dimensions = np.roll(trailing_dimensions, 1, axis=0)
        trailing_dimensions[0] = dimensions

        # Apply weighted moving average over the last SMOOTH_NUM_FRAMES frames
        dimensions = np.sum(
            trailing_dimensions * SMOOTHING_WEIGHTS_NORMALIZED, axis=0
        ).astype(int)

        # Write the frame
        shared_state.frame = opencv_image[
            dimensions[0] : dimensions[1], dimensions[2] : dimensions[3]
        ]

        if DEBUG_LEVEL == 0:
            return
        elif DEBUG_LEVEL == 1:
            # Draw bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(opencv_image, start_point, end_point, TEXT_COLOR, 3)

        elif DEBUG_LEVEL == 2:
            height, width, _ = opencv_image.shape
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(opencv_image, start_point, end_point, TEXT_COLOR, 3)

                # Draw keypoints
                for keypoint in detection.keypoints:
                    keypoint_px = _normalized_to_pixel_coordinates(
                        keypoint.x, keypoint.y, width, height
                    )
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(opencv_image, keypoint_px, radius, color, thickness)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = "" if category_name is None else category_name
                probability = round(category.score, 2)
                result_text = f"{category_name} ({probability})"
                text_location = (
                    MARGIN + bbox.origin_x,
                    MARGIN + ROW_SIZE + bbox.origin_y,
                )
                cv2.putText(
                    opencv_image,
                    result_text,
                    text_location,
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                )

    return visualize, shared_state


def main():
    print("Press 'q' to exit")
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Initialize the visuazlier with the captured constants
    visualizer, state = make_visualizer()

    # Initialize the MediaPipe Face Detector in live stream mode
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=visualizer,
    )

    # Capture the webcam feed
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        exit("Failed to open camera")
    cv2.namedWindow("Video", flags=(cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))
    with FaceDetector.create_from_options(options) as mp_face_detection:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                exit("Failed to get frame from webcam")

            # convert to MP image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # compute timestamp
            ms = time.time_ns() // 1_000_000

            # Detect faces in the frame, writing to global
            mp_face_detection.detect_async(mp_image, ms)

            # Draw frame if it's available
            if state.frame is not None:
                cv2.imshow("Video", state.frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
