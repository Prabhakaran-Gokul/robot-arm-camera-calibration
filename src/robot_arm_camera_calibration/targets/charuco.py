from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from robot_arm_camera_calibration.cameras.base import CameraIntrinsics
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.targets.base import CalibrationTarget, TargetDetection

_MIN_CORNERS_FOR_POSE = 4


class CharucoBoardTarget(CalibrationTarget):
    def __init__(
        self,
        dictionary_name: str,
        squares_x: int,
        squares_y: int,
        square_length_m: float,
        marker_length_m: float,
    ) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        self._board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length_m, marker_length_m, dictionary
        )
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        intrinsics: CameraIntrinsics,
        *,
        annotate: bool = False,
    ) -> TargetDetection | None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detectBoard(image)
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < _MIN_CORNERS_FOR_POSE
        ):
            return None

        object_points, image_points = self._board.matchImagePoints(
            list(charuco_corners), charuco_ids
        )
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        if not ok:
            return None

        reprojected, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        rms_px = float(
            np.sqrt(
                np.mean(
                    np.sum((reprojected.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1)
                )
            )
        )

        annotated = None
        if annotate:
            annotated = cv2.aruco.drawDetectedMarkers(image.copy(), marker_corners, marker_ids)
            cv2.drawFrameAxes(
                annotated, intrinsics.camera_matrix, intrinsics.dist_coeffs, rvec, tvec, 0.05
            )
            annotated = annotated.astype(np.uint8)

        return TargetDetection(
            pose_in_camera=Transform.from_rvec_tvec(rvec, tvec),
            reprojection_rms_px=rms_px,
            annotated_image=annotated,
        )
