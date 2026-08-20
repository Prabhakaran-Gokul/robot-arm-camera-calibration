"""Renders a marker/board at a known pose into a synthetic camera image and
checks detection recovers that pose, without needing a printed target or a
real camera.

Note on the rotation used: OpenCV's camera convention has +Y pointing down
in the image, so a target held upright and facing the camera corresponds to
roughly a 180 degree rotation about X, not the identity rotation — using a
near-zero rvec here would render a mirrored (and therefore undetectable)
target.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm_camera_calibration.cameras.base import CameraIntrinsics
from robot_arm_camera_calibration.core.transform import Transform
from robot_arm_camera_calibration.targets.aruco import ArucoMarkerTarget, _square_object_points
from robot_arm_camera_calibration.targets.charuco import CharucoBoardTarget

_INTRINSICS = CameraIntrinsics(
    width=640, height=480, fx=800.0, fy=800.0, cx=320.0, cy=240.0, dist_coeffs=np.zeros(5)
)
_FACING_CAMERA_ROTATION = Rotation.from_euler("x", 180, degrees=True)


def _facing_camera_rvec(tilt_deg: tuple[float, float, float]) -> np.ndarray:
    tilt = Rotation.from_euler("xyz", tilt_deg, degrees=True)
    return (tilt * _FACING_CAMERA_ROTATION).as_rotvec()


def _warp_planar_image(
    planar_image: np.ndarray,
    object_corners_pixel_order: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Warps a top-down (pixel-row-0-at-top) rendering of a planar target into a synthetic
    camera view at the given pose. `object_corners_pixel_order` must list the target's 3D
    corners (object frame) in the same order as the image's [TL, TR, BR, BL] pixel corners."""
    height, width = planar_image.shape[:2]
    image_points, _ = cv2.projectPoints(
        object_corners_pixel_order, rvec, tvec, _INTRINSICS.camera_matrix, _INTRINSICS.dist_coeffs
    )
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(src, image_points.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(
        planar_image,
        homography,
        (_INTRINSICS.width, _INTRINSICS.height),
        borderValue=(255, 255, 255),
        flags=cv2.INTER_AREA,
    )


def _marker_image(dictionary_name: str, marker_id: int, side_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.cvtColor(bits, cv2.COLOR_GRAY2BGR)


def test_aruco_target_recovers_known_pose() -> None:
    dictionary_name, marker_id, marker_length_m = "DICT_5X5_100", 3, 0.06
    marker_image = _marker_image(dictionary_name, marker_id, side_px=300)

    true_rvec = _facing_camera_rvec((8, 12, 5))
    true_tvec = np.array([0.03, -0.02, 0.45])
    object_corners = _square_object_points(marker_length_m)
    canvas = _warp_planar_image(marker_image, object_corners, true_rvec, true_tvec)

    target = ArucoMarkerTarget(dictionary_name, marker_id, marker_length_m)
    detection = target.detect(canvas, _INTRINSICS)

    assert detection is not None
    true_pose = Transform.from_rvec_tvec(true_rvec, true_tvec)
    np.testing.assert_allclose(
        detection.pose_in_camera.translation, true_pose.translation, atol=0.005
    )
    assert detection.reprojection_rms_px < 1.0


def test_aruco_target_returns_none_when_absent() -> None:
    target = ArucoMarkerTarget("DICT_5X5_100", 3, 0.06)
    blank = np.full((480, 640, 3), 255, np.uint8)
    assert target.detect(blank, _INTRINSICS) is None


_CHARUCO_DICT = "DICT_5X5_100"
_CHARUCO_SQUARES_X, _CHARUCO_SQUARES_Y = 5, 7
_CHARUCO_SQUARE_M, _CHARUCO_MARKER_M = 0.03, 0.022
_CHARUCO_PX_PER_SQUARE = 40


def _charuco_board_and_image() -> tuple[cv2.aruco.CharucoBoard, np.ndarray, np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _CHARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (_CHARUCO_SQUARES_X, _CHARUCO_SQUARES_Y), _CHARUCO_SQUARE_M, _CHARUCO_MARKER_M, dictionary
    )
    board_image = board.generateImage(
        (_CHARUCO_SQUARES_X * _CHARUCO_PX_PER_SQUARE, _CHARUCO_SQUARES_Y * _CHARUCO_PX_PER_SQUARE),
        marginSize=0,
    )
    board_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    board_object_corners = np.array(
        [
            [0, 0, 0],
            [_CHARUCO_SQUARES_X * _CHARUCO_SQUARE_M, 0, 0],
            [
                _CHARUCO_SQUARES_X * _CHARUCO_SQUARE_M,
                _CHARUCO_SQUARES_Y * _CHARUCO_SQUARE_M,
                0,
            ],
            [0, _CHARUCO_SQUARES_Y * _CHARUCO_SQUARE_M, 0],
        ],
        dtype=np.float64,
    )
    return board, board_image, board_object_corners


def test_charuco_target_recovers_known_pose() -> None:
    _board, board_image, board_object_corners = _charuco_board_and_image()

    # Unlike the single-marker convention, CharucoBoard's object frame has +Y already
    # aligned with the image's row direction, so no 180-degree "facing camera" flip is needed.
    true_rvec = Rotation.from_euler("xyz", [10, -8, 3], degrees=True).as_rotvec()
    true_tvec = np.array([-0.02, 0.03, 0.55])
    canvas = _warp_planar_image(board_image, board_object_corners, true_rvec, true_tvec)

    target = CharucoBoardTarget(
        _CHARUCO_DICT, _CHARUCO_SQUARES_X, _CHARUCO_SQUARES_Y, _CHARUCO_SQUARE_M, _CHARUCO_MARKER_M
    )
    detection = target.detect(canvas, _INTRINSICS)

    assert detection is not None
    true_pose = Transform.from_rvec_tvec(true_rvec, true_tvec)
    np.testing.assert_allclose(
        detection.pose_in_camera.translation, true_pose.translation, atol=0.005
    )
    assert detection.reprojection_rms_px < 1.0


def test_charuco_target_returns_none_when_absent() -> None:
    target = CharucoBoardTarget(
        _CHARUCO_DICT, _CHARUCO_SQUARES_X, _CHARUCO_SQUARES_Y, _CHARUCO_SQUARE_M, _CHARUCO_MARKER_M
    )
    blank = np.full((480, 640, 3), 255, np.uint8)
    assert target.detect(blank, _INTRINSICS) is None


def test_charuco_target_handles_partially_visible_board_without_raising() -> None:
    """Regression test: a board sliding out of frame yields a corner set that's technically
    non-empty but too small/degenerate for cv2.solvePnP's default (DLT) algorithm, which raises
    a C++ exception rather than failing gracefully for fewer than 6 points. detect() must catch
    this and return None, not propagate the exception into the caller's update loop."""
    _board, board_image, _corners = _charuco_board_and_image()
    _height, width = board_image.shape[:2]
    target = CharucoBoardTarget(
        _CHARUCO_DICT, _CHARUCO_SQUARES_X, _CHARUCO_SQUARES_Y, _CHARUCO_SQUARE_M, _CHARUCO_MARKER_M
    )

    for row_height in (100, 60, 30):
        canvas = np.full((480, 640, 3), 255, np.uint8)
        canvas[100 : 100 + row_height, 50 : 50 + width] = board_image[:row_height, :]
        assert target.detect(canvas, _INTRINSICS) is None
