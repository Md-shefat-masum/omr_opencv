from __future__ import annotations

import cv2
import numpy as np


def _is_corner_square_candidate(cnt: np.ndarray) -> tuple[bool, dict[str, float]]:
    """
    Returns (is_candidate, features). Uses fast shape features that separate
    filled corner squares from the filled answer bubbles (circles).
    """
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return False, {}

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return False, {}

    aspect = w / float(h)
    bbox_area = float(w * h)
    solidity = area / (bbox_area + 1e-6)  # square ~1.0, circle ~0.785

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0
    extent = area / (rect_area + 1e-6)  # less sensitive to rotation

    # Size range for your sheet after resize(550x820)
    if not (5 <= w <= 80 and 5 <= h <= 80):
        return False, {
            "area": area,
            "w": float(w),
            "h": float(h),
            "aspect": aspect,
            "solidity": solidity,
            "extent": extent,
        }

    if not (0.80 <= aspect <= 1.25):
        return False, {
            "area": area,
            "w": float(w),
            "h": float(h),
            "aspect": aspect,
            "solidity": solidity,
            "extent": extent,
        }

    # Distinguish squares from circles: circle extent~0.785, square extent~1.0
    if extent < 0.86:
        return False, {
            "area": area,
            "w": float(w),
            "h": float(h),
            "aspect": aspect,
            "solidity": solidity,
            "extent": extent,
        }

    return True, {
        "area": area,
        "w": float(w),
        "h": float(h),
        "aspect": aspect,
        "solidity": solidity,
        "extent": extent,
    }


def _outermost_corner_for_marker(box_pts: np.ndarray, corner_xy: tuple[float, float]) -> np.ndarray:
    cx, cy = corner_xy
    d = np.sum((box_pts - np.array([cx, cy], dtype=np.float32)) ** 2, axis=1)
    return box_pts[int(np.argmin(d))]


def _detect_corner_markers(resized_bgr: np.ndarray) -> dict[str, np.ndarray]:
    h, w = resized_bgr.shape[:2]
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    corners = {
        "tl": (0.0, 0.0),
        "tr": (float(w - 1), 0.0),
        "bl": (0.0, float(h - 1)),
        "br": (float(w - 1), float(h - 1)),
    }

    # Search each corner region independently; avoids being confused by bubbles/text.
    roi_w = int(0.25 * w)
    roi_h = int(0.25 * h)
    kernel = np.ones((3, 3), np.uint8)

    def _corner_roi(key: str) -> tuple[int, int, int, int]:
        if key == "tl":
            return 0, 0, roi_w, roi_h
        if key == "tr":
            return w - roi_w, 0, roi_w, roi_h
        if key == "bl":
            return 0, h - roi_h, roi_w, roi_h
        return w - roi_w, h - roi_h, roi_w, roi_h

    pts: dict[str, np.ndarray] = {}
    for key, corner_xy in corners.items():
        x0, y0, rw, rh = _corner_roi(key)
        g = gray[y0 : y0 + rh, x0 : x0 + rw]

        thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = None
        best_box = None

        for cnt in contours:
            ok, feats = _is_corner_square_candidate(cnt)
            if not ok:
                continue

            rect = cv2.minAreaRect(cnt)
            (ccx, ccy), (bw, bh), _ = rect
            if bw <= 0 or bh <= 0:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)

            # Convert box + centroid to full-image coordinates
            box[:, 0] += x0
            box[:, 1] += y0
            cx_full = ccx + x0
            cy_full = ccy + y0

            area = feats["area"]
            dist2 = (cx_full - corner_xy[0]) ** 2 + (cy_full - corner_xy[1]) ** 2
            score = dist2 / (area + 1e-6)
            if best_score is None or score < best_score:
                best_score = float(score)
                best_box = box

        if best_box is None:
            raise RuntimeError(
                f"Could not find corner marker in corner '{key}'. "
                "Try adjusting ROI size/thresholding/shape filters."
            )

        pts[key] = _outermost_corner_for_marker(best_box, corner_xy)

    return pts

