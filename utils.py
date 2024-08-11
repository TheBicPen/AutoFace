from mediapipe.tasks.python.components.containers import BoundingBox, Category
from mediapipe.tasks.python.components.containers.detections import (
    Detection,
    DetectionResult,
)


def detection_result_from_str(detections_str: str) -> DetectionResult:
    """
    Parse a string to get detection results

    >>> detection_result_from_str("0,0,100,100,0.5")
    DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=0, origin_y=0, width=100, height=100), categories=[Category(index=None, score=0.5, display_name=None, category_name=None)], keypoints=None)])

    >>> d2 = detection_result_from_str("0,0,200,200,0.1;400,400,100,100,0.8")
    >>> len(d2.detections)
    2
    >>> d2.detections[0]
    Detection(bounding_box=BoundingBox(origin_x=0, origin_y=0, width=200, height=200), categories=[Category(index=None, score=0.1, display_name=None, category_name=None)], keypoints=None)
    >>> d2.detections[1]
    Detection(bounding_box=BoundingBox(origin_x=400, origin_y=400, width=100, height=100), categories=[Category(index=None, score=0.8, display_name=None, category_name=None)], keypoints=None)
    """
    if not detections_str:
        return DetectionResult([])
    return DetectionResult(
        [
            Detection(
                bounding_box=BoundingBox(*(int(x) for x in detection_str.split(",")[:-1])), categories=[Category(score=float(detection_str.split(",")[-1]))]
            )
            for detection_str in detections_str.split(";")
        ]
    )


def detection_result_to_str(detection_result: DetectionResult):
    """
    >>> d = DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=0, origin_y=0, width=100, height=100), categories=[Category(index=None, score=0.5, display_name=None, category_name=None)], keypoints=None)])
    >>> detection_result_to_str(d)
    '0,0,100,100,0.5'
    """
    return ";".join(
        f"{d.bounding_box.origin_x},{d.bounding_box.origin_y},{d.bounding_box.width},{d.bounding_box.height},{d.categories[0].score}"
        for d in detection_result.detections
    )
