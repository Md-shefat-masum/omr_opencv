from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CoachingCodeResult:
    coaching_roi: np.ndarray
    coaching_bin: np.ndarray
    first_row: np.ndarray
    first_row_bin: np.ndarray
    first_row_edges: np.ndarray
    filled_by_col: list[list[int]]
    filled_numbers: list[int]


def _adaptive_bin(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    b = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return b


def get_coaching_code(
    *,
    resized_bgr: np.ndarray,
    out_bgr: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    cols: int,
    rows: int,
    grid_color_bgr: tuple[int, int, int] = (255, 0, 0),
    fill_threshold: float = 0.28,
    fill_threshold_inner: float | None = None,
    fill_threshold_top_left: float | None = None,
    draw_grid: bool = True,
    draw_box: bool = True,
    draw_filled_marks: bool = True,
    draw_detection_circles: bool = True,
) -> CoachingCodeResult:
    """
    Coaching code block:
    - Draw box + grid (optional)
    - Extract first row (digits) + edges previews
    - Detect filled circles for remaining rows (skip first row)
    - Return filled_by_col and filled_numbers (handles multi-fills)
    """
    if draw_box:
        coaching_square = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
        cv2.polylines(out_bgr, [coaching_square], isClosed=True, color=(0, 0, 255), thickness=1)

    if draw_grid:
        for c in range(1, cols):
            x = int(round(x_min + (x_max - x_min) * c / cols))
            cv2.line(out_bgr, (x, y_min), (x, y_max), grid_color_bgr, 1)
        for r in range(1, rows):
            y = int(round(y_min + (y_max - y_min) * r / rows))
            cv2.line(out_bgr, (x_min, y), (x_max, y), grid_color_bgr, 1)

    coaching_roi = resized_bgr[y_min : y_max + 1, x_min : x_max + 1]
    coaching_gray = cv2.cvtColor(coaching_roi, cv2.COLOR_BGR2GRAY)
    coaching_bin = _adaptive_bin(coaching_gray)

    # First row (exactly one row high)
    row_h = int(round((y_max - y_min) / rows))
    first_row_h = min(coaching_roi.shape[0], max(1, row_h))
    first_row = coaching_roi[0:first_row_h, :]
    first_row_gray = cv2.cvtColor(first_row, cv2.COLOR_BGR2GRAY)

    # Improve digits visibility
    first_row_up = cv2.resize(first_row_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    first_row_eq = clahe.apply(first_row_up)
    first_row_bin = cv2.adaptiveThreshold(
        cv2.GaussianBlur(first_row_eq, (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    first_row_bin = cv2.morphologyEx(first_row_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    first_row_bin = cv2.morphologyEx(first_row_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    first_row_edges = cv2.Canny(first_row_bin, 30, 120)

    # Filled circles detection (skip first row)
    cell_w = coaching_roi.shape[1] / float(cols)
    cell_h = coaching_roi.shape[0] / float(rows)
    filled_by_col: list[list[int]] = [[] for _ in range(cols)]

    for c in range(cols):
        for r in range(1, rows):
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))

            cell = coaching_bin[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            ch, cw = cell.shape[:2]
            cx, cy = cw // 2, ch // 2
            radius_extra_px = 5
            radius = int(0.32 * min(cw, ch)) + radius_extra_px
            radius = min(radius, cx, cy, (cw - 1 - cx), (ch - 1 - cy))
            if radius <= 1:
                continue

            if draw_detection_circles:
                center_x = x_min + x0 + cx
                center_y = y_min + y0 + cy
                cv2.circle(
                    out_bgr,
                    (int(center_x), int(center_y)),
                    radius,
                    (160, 160, 160),
                    1,
                )

            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            ink = int(np.sum(cell[mask == 255] > 0))
            tot = int(np.sum(mask == 255))
            ratio = ink / float(tot + 1e-6)

            inner_shrink_px = 4
            inner_radius = max(1, radius - inner_shrink_px)
            inner_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(inner_mask, (cx, cy), inner_radius, 255, -1)
            inner_ink = int(np.sum(cell[inner_mask == 255] > 0))
            inner_tot = int(np.sum(inner_mask == 255))
            ratio_inner = inner_ink / float(inner_tot + 1e-6)

            # Some marks touch the bubble on the top-left; prioritize that region too.
            tl_dx = int(round(radius * 0.35))
            tl_dy = int(round(radius * 0.35))
            tl_r = max(1, int(round(radius * 0.55)))
            tl_cx = int(np.clip(cx - tl_dx, 0, cw - 1))
            tl_cy = int(np.clip(cy - tl_dy, 0, ch - 1))
            tl_r = min(tl_r, tl_cx, tl_cy, (cw - 1 - tl_cx), (ch - 1 - tl_cy))
            ratio_tl = 0.0
            if tl_r > 1:
                tl_mask = np.zeros((ch, cw), dtype=np.uint8)
                cv2.circle(tl_mask, (tl_cx, tl_cy), tl_r, 255, -1)
                tl_ink = int(np.sum(cell[tl_mask == 255] > 0))
                tl_tot = int(np.sum(tl_mask == 255))
                ratio_tl = tl_ink / float(tl_tot + 1e-6)

            thr_inner = fill_threshold_inner if fill_threshold_inner is not None else (fill_threshold * 0.75)
            thr_tl = fill_threshold_top_left if fill_threshold_top_left is not None else (fill_threshold * 0.60)

            row_index_0_to_9 = r - 1
            if (ratio >= fill_threshold) or (ratio_inner >= thr_inner) or (ratio_tl >= thr_tl):
                filled_by_col[c].append(row_index_0_to_9)
                if draw_filled_marks:
                    center_x = x_min + x0 + cx
                    center_y = y_min + y0 + cy
                    cv2.circle(
                        out_bgr,
                        (int(center_x), int(center_y)),
                        radius,
                        (0, 255, 255),
                        2,
                    )

    # Build numbers from "last value per column", but keep alternates if multi-filled
    col_choices: list[list[int]] = []
    for col_vals in filled_by_col:
        if not col_vals:
            col_choices.append([0])
        else:
            uniq = sorted(set(col_vals))
            last = col_vals[-1]
            ordered = [last] + [v for v in uniq if v != last]
            col_choices.append(ordered)

    filled_numbers: list[int] = []
    for digits in itertools.product(*col_choices):
        filled_numbers.append(int("".join(str(d) for d in digits)))

    return CoachingCodeResult(
        coaching_roi=coaching_roi,
        coaching_bin=coaching_bin,
        first_row=first_row,
        first_row_bin=first_row_bin,
        first_row_edges=first_row_edges,
        filled_by_col=filled_by_col,
        filled_numbers=filled_numbers,
    )

