from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.cameras.base import CameraIntrinsics
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.targets.base import CalibrationTarget, TargetDetection


def _square_object_points(side_length_m: float) -> npt.NDArray[np.float64]:
    half = side_length_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
    )


class ArucoMarkerTarget(CalibrationTarget):
    def __init__(self, dictionary_name: str, marker_id: int, marker_length_m: float) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self._marker_id = marker_id
        self._object_points = _square_object_points(marker_length_m)

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
        *,
        annotate: bool = False,
    ) -> TargetDetection | None:
        corners, ids, _rejected = self._detector.detectMarkers(image)
        if ids is None:
            return None
        matches = np.flatnonzero(ids.flatten() == self._marker_id)
        if matches.size == 0:
            return None

        image_points = corners[int(matches[0])].reshape(4, 2)
        # A severely foreshortened/edge-clipped marker can yield corners OpenCV raises a C++
        # exception for rather than just returning ok=False, so it must be caught here rather
        # than left to propagate and kill the caller's update loop.
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return None
        if not ok:
            return None

        reprojected, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        rms_px = float(
            np.sqrt(np.mean(np.sum((reprojected.reshape(4, 2) - image_points) ** 2, axis=1)))
        )

        annotated = None
        if annotate:
            annotated = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            cv2.drawFrameAxes(
                annotated, intrinsics.camera_matrix, intrinsics.dist_coeffs, rvec, tvec, 0.03
            )
            annotated = annotated.astype(np.uint8)

        return TargetDetection(
            pose_in_camera=Transform.from_rvec_tvec(rvec, tvec),
            reprojection_rms_px=rms_px,
            annotated_image=annotated,
        )
