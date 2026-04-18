"""
Perspective-normalize a resized OMR scan to a fixed template quad (550×820).

Flow: rough resize → detect four anchor corners → warp so they land on canonical
template coordinates. Downstream ROI code then sees a flat, aligned sheet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

_omr_dir = Path(__file__).resolve().parent.parent
if str(_omr_dir) not in sys.path:
    sys.path.insert(0, str(_omr_dir))

from check_4_square import refine_four_corners

# Canonical canvas size (matches existing pipeline).
OMR_CANVAS_SIZE_WH: tuple[int, int] = (550, 820)

# Fixed destination quad on the canvas: tl, tr, br, bl (aligned “perfect” form).
OMR_TEMPLATE_CORNERS: dict[str, tuple[float, float]] = {
    "tl": (50.0, 50.0),
    "tr": (508.0, 50.0),
    "br": (508.0, 775.0),
    "bl": (50.0, 775.0),
}

# Seeds for corner refinement on the resized image (usually same as template).
OMR_CORNER_SEEDS: dict[str, tuple[float, float]] = {
    "tl": (50.0, 50.0),
    "tr": (508.0, 50.0),
    "br": (508.0, 775.0),
    "bl": (50.0, 775.0),
}


def preprocess_gray_for_corners(bgr: np.ndarray) -> np.ndarray:
    """Same preprocessing as ``check_4_square`` (blur before refinement)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _quad_stack(
    corners: dict[str, tuple[float, float] | np.ndarray],
    order: tuple[str, str, str, str] = ("tl", "tr", "br", "bl"),
) -> np.ndarray:
    return np.array(
        [np.asarray(corners[k], dtype=np.float64).reshape(2) for k in order],
        dtype=np.float32,
    )


def warp_resized_scan_to_template(
    resized_bgr: np.ndarray,
    *,
    corner_seeds: dict[str, tuple[float, float]] | None = None,
    template_corners: dict[str, tuple[float, float]] | None = None,
    search_radius: int = 95,
    border_value: tuple[int, int, int] = (255, 255, 255),
    br_mode: Literal["refined", "parallelogram", "blend"] = "parallelogram",
    br_blend: float = 0.5,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Map the four detected anchor corners on ``resized_bgr`` onto the fixed template quad.

    Parameters
    ----------
    resized_bgr
        Already resized to ``OMR_CANVAS_SIZE_WH`` (e.g. 550×820).
    corner_seeds
        Initial guesses for ``refine_four_corners``; default ``OMR_CORNER_SEEDS``.
    template_corners
        Destination quad; default ``OMR_TEMPLATE_CORNERS``.
    br_mode, br_blend
        Passed to ``refine_four_corners`` for bottom-right: default ``parallelogram`` sets
        ``br = tr + bl - tl`` so the frame matches the bottom and right edges implied by the
        other corners (same as the green overlay L-corner in a perfect parallelogram).

    Returns
    -------
    warped_bgr
        Same size as input; perspective-corrected so markers sit on template coords.
    refined_src
        Refined source corners (float32 arrays), keys tl, tr, br, bl.
    perspective_3x3
        3×3 homography mapping source → destination (pixel coords).
    """
    tw, th = OMR_CANVAS_SIZE_WH
    h, w = resized_bgr.shape[:2]
    if (w, h) != (tw, th):
        raise ValueError(
            f"Expected image size {tw}×{th}, got {w}×{h}. Resize before warping."
        )

    seeds = corner_seeds if corner_seeds is not None else OMR_CORNER_SEEDS
    dst_map = template_corners if template_corners is not None else OMR_TEMPLATE_CORNERS

    gray = preprocess_gray_for_corners(resized_bgr)
    refined = refine_four_corners(
        gray,
        seeds,
        search_radius=search_radius,
        br_mode=br_mode,
        br_blend=br_blend,
    )
    refined_src: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float32) for k, v in refined.items()}

    src = _quad_stack(refined_src)
    dst = _quad_stack({k: dst_map[k] for k in ("tl", "tr", "br", "bl")})

    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        resized_bgr,
        m,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return warped, refined_src, m
