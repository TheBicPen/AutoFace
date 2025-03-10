#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace, FileType
from pathlib import Path
import sys
import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, Union
import math
import time

from utils import detection_result_to_str

def parse_args():
    # Default values
    SMOOTH_NUM_FRAMES = 10  # Number of frames to average over
    LOW_PASS_THRESHOLD = 20  # Changes smaller than this (in pixels) are ignored
    INITIAL_SIZE = 400

    # Gaps between the edge of the face and the edge of the window in pixels
    MARGIN_TOP = 75
    MARGIN_BOTTOM = 25
    MARGIN_LEFT = 50
    MARGIN_RIGHT = 50

    # Minimum size of the window
    MIN_WIDTH = None
    MIN_HEIGHT = None

    parser = ArgumentParser()
    parser.add_argument(
        "-o", "--log-file",
        type=FileType('w'),
        help="Log the detection locations to a file. Useful for debugging"
    )
    parser.add_argument(
        "--debug",
        type=int,
        choices=range(4),
        default=0,
        help="1: Show bounding boxes for detected face. 2: Also show confidence level and other detected faces. 3: Draw bounding box instead of cropping the window to it",
    )
    parser.add_argument(
        "--low-pass-threshold",
        type=int,
        default=LOW_PASS_THRESHOLD,
        help="Movements smaller than this number (in pixels) are ignored",
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=INITIAL_SIZE,
        help="Initial window size (will smoothly scale to actual size)",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=SMOOTH_NUM_FRAMES,
        help="Number of frames to smoothly apply changes",
    )
    parser.add_argument(
        "--margin-top",
        type=int,
        default=MARGIN_TOP,
        help="Gap between the top of the face and the edge of the window",
    )
    parser.add_argument(
        "--margin-bottom",
        type=int,
        default=MARGIN_BOTTOM,
        help="Gap between the bottom of the face and the edge of the window",
    )
    parser.add_argument(
        "--margin-left",
        type=int,
        default=MARGIN_LEFT,
        help="Gap between the left of the face and the edge of the window",
    )
    parser.add_argument(
        "--margin-right",
        type=int,
        default=MARGIN_RIGHT,
        help="Gap between the right of the face and the edge of the window",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=MIN_WIDTH,
        help="Minimum width of the window",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=MIN_HEIGHT,
        help="Minimum height of the window",
    )
    parser.add_argument(
        "--tracking-model",
        default=Path(__file__).parent / "blaze_face_short_range.tflite",
        help="Path to the MediaPipe face detection model",
    )
    return parser.parse_args()


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
        self.frame: None | np.ndarray = None
        self.frozen = False
        # Re-center the frame immediately. Reset this each frame
        self.do_flush = False


def make_visualizer(args: Namespace):
    # Mutable state captured inside visualize
    shared_state = State()
    # Start with a fixed initial size
    trailing_dimensions = np.array([[0, args.initial_size] * 2] * args.smooth_frames)
    low_pass_last_amounts = np.array([0, args.initial_size] * 2)

    # Constants
    # Normalization factor
    SMOOTH_TRIANGLE = args.smooth_frames * (args.smooth_frames + 1) / 2
    # Triangular weighted moving average - weights n, n-1, ... 2, 1
    SMOOTHING_WEIGHTS = (
        np.tile(np.arange(args.smooth_frames, 0, -1), 4)
        .reshape(4, args.smooth_frames)
        .transpose()
    )
    # Normalize weights
    SMOOTHING_WEIGHTS_NORMALIZED = SMOOTHING_WEIGHTS / SMOOTH_TRIANGLE
    assert np.array_equal(
        np.sum(SMOOTHING_WEIGHTS_NORMALIZED, axis=0), [1.0] * 4
    ), "Failed to normalize weights"

    # Constants for drawing debug info on screen
    TEXT_MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # blue - the format is BGR
    VIEWPORT_BOX_COLOR = (10, 200, 10)  # BGR
    MARGIN_BOX_COLOR = (10, 10, 200)  # BGR

    def visualize(detection_result, image, timestamp) -> None:
        """
        Crops the image to the bounding box, drawing bounding boxes and keypoints
        on the input image based on the DEBUG_LEVEL
        """
        opencv_image = np.array(image.numpy_view())
        image_height, image_width, _ = opencv_image.shape

        if args.log_file:
            args.log_file.write(f"{timestamp}:{detection_result_to_str(detection_result)}\n")

        # Choose detecton with the highest probability
        max_probability = 0
        max_detection = None
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.score > max_probability:
                max_probability = category.score
                max_detection = detection

        nonlocal shared_state
        nonlocal trailing_dimensions
        no_detection_processing = (max_detection is None or shared_state.frozen) and not shared_state.do_flush
        if no_detection_processing:
            # If no face was detected, use the new frame with the previous dimensions
            dimensions = trailing_dimensions[0]
        else:
            bbox = max_detection.bounding_box # type: ignore - I don't know why it doesn't understand that this is not None

            # Clamp margin to edges of frame
            start_x = max(bbox.origin_x - args.margin_left, 0)
            end_x = min(bbox.origin_x + bbox.width + args.margin_right, image_width)
            # Pad width on both sides to reach minimum
            missing_width = (args.min_width - (end_x - start_x)) if args.min_width else 0
            if missing_width > 0:
                pad_left = min(missing_width // 2, start_x)
                start_x -= pad_left
                missing_width -= pad_left
                end_x += missing_width

            # Clamp margin to edges of frame
            start_y = max(bbox.origin_y - args.margin_top, 0)
            end_y = min(bbox.origin_y + bbox.height + args.margin_bottom, image_height)
            # Pad height on both sides to reach minimum
            missing_height = (args.min_height - (end_y - start_y)) if args.min_height else 0
            if missing_height > 0:
                pad_top = min(missing_height // 2, start_y)
                start_y -= pad_top
                missing_height -= pad_top
                end_y += missing_height

            current_box = np.array([start_y, end_y, start_x, end_x])

            # Ignore changes below a threshold
            nonlocal low_pass_last_amounts
            if shared_state.do_flush or (
                np.max(np.abs(current_box - low_pass_last_amounts))
                >= args.low_pass_threshold
            ):
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

        shared_state.do_flush = False

        if args.debug == 3:
            shared_state.frame = opencv_image
            # Dimensions array is: [start y, end y, start x, end x]
            cv2.rectangle(opencv_image, (dimensions[2], dimensions[0]), (dimensions[3], dimensions[1]), TEXT_COLOR, 3)
            return

        if no_detection_processing:
            # If nothing was detected, we have no useful debug information
            return

        if args.debug == 0:
            return
        elif args.debug == 1:
            # Draw bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(opencv_image, start_point, end_point, TEXT_COLOR, 3)

        elif args.debug == 2:
            # Draw viewport rectangle before smoothing
            cv2.rectangle(opencv_image, (start_x, start_y), (end_x, end_y), VIEWPORT_BOX_COLOR, 3)

            margin_start = (bbox.origin_x - args.margin_left, bbox.origin_y - args.margin_top)
            margin_end = (bbox.origin_x + bbox.width + args.margin_right, bbox.origin_y + bbox.height + args.margin_bottom)
            # Draw margin rectangle - may be partially off-screen
            cv2.rectangle(opencv_image, margin_start, margin_end, MARGIN_BOX_COLOR, 3)
            # print(margin_start, margin_end)

            # Draw every detected face
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                # print(start_point, end_point)
                cv2.rectangle(opencv_image, start_point, end_point, TEXT_COLOR, 3)

                if detection.keypoints:
                    # Draw keypoints
                    for keypoint in detection.keypoints:
                        keypoint_px = _normalized_to_pixel_coordinates(
                            keypoint.x, keypoint.y, image_width, image_height
                        )
                    color, thickness, radius = (0, 255, 0), 2, 2
                    # This signature seems correct - I don't know why the type checker complains
                    cv2.circle(opencv_image, keypoint_px, radius, color, thickness) # type: ignore

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = "" if category_name is None else category_name
                probability = round(category.score, 2)
                result_text = f"{category_name} ({probability})"
                text_location = (
                    TEXT_MARGIN + bbox.origin_x,
                    TEXT_MARGIN + ROW_SIZE + bbox.origin_y,
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
    # Probably runs on 3.9 too, but 3.10 is very widely used
    assert sys.version_info >= (3, 10), "This tool requires at least python 3.10"
    args = parse_args()
    print("Press 'q' to exit, 'f' to freeze the window size, arrow keys to resize")

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Initialize the visuazlier with the captured constants
    visualizer, state = make_visualizer(args)

    # Initialize the MediaPipe Face Detector in live stream mode
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.tracking_model),
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

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Press 'q' to exit the loop
                break
            elif key & 0xFF == ord("f"):
                state.frozen = not state.frozen
            elif key == 81:
                # left arrow
                args.margin_left -= 5
                args.margin_right -= 5
                state.do_flush = True
            elif key == 82:
                # up arrow
                args.margin_top += 5
                args.margin_bottom += 5
                state.do_flush = True
            elif key == 83:
                # right arrow
                args.margin_left += 5
                args.margin_right += 5
                state.do_flush = True
            elif key == 84:
                # down arrow
                args.margin_top -= 5
                args.margin_bottom -= 5
                state.do_flush = True
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
