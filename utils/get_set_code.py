from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SetCodeResult:
    """SET detection outputs; ``set_debug_bgr`` is the ROI with yellow/green scan rings only."""

    set_roi: np.ndarray
    set_bin: np.ndarray
    set_label: str
    set_crop_view: np.ndarray
    set_debug_bgr: np.ndarray | None


def get_set_code(
    *,
    resized_bgr: np.ndarray,
    out_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    rows: int = 6,
    skip_first_row: bool = True,
    fill_threshold: float = 0.25,
    crop_pad: int = 8,
    crop_view_size: tuple[int, int] = (300, 600),
    debug_visual: bool = True,
    debug_print: bool = False,
) -> SetCodeResult:
    """
    SET block:
    - Draw yellow border and red row boxes (rows distributed dynamically across height)
    - Detect which of rows 2..6 are filled -> label a..e (skip first row with letters)
    - Return zoom crop for plotting.

    When ``debug_visual`` is True, draws on ``out_bgr`` and builds ``set_debug_bgr``:
    - Yellow: scanned circle (same mask used for fill ratio)
    - Green: selected row (ratio >= threshold)

    When ``debug_print`` is True, prints one line per scanned row to stdout.
    """
    cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow border

    h = max(1, y2 - y1)
    for i in range(rows):
        y_top = y1 + int(round(i * h / rows))
        y_bot = y1 + int(round((i + 1) * h / rows))
        cv2.rectangle(out_bgr, (x1, y_top), (x2, y_bot), (0, 0, 255), 1)  # red row boxes

    set_roi = resized_bgr[y1 : y2 + 1, x1 : x2 + 1]
    set_gray = cv2.cvtColor(set_roi, cv2.COLOR_BGR2GRAY)
    set_blur = cv2.GaussianBlur(set_gray, (5, 5), 0)
    set_bin = cv2.adaptiveThreshold(
        set_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    set_bin = cv2.morphologyEx(set_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Row boundaries in bin space
    row_edges: list[tuple[int, int]] = []
    for i in range(rows):
        y_top = int(round(i * set_bin.shape[0] / rows))
        y_bot = int(round((i + 1) * set_bin.shape[0] / rows))
        row_edges.append((y_top, y_bot))

    start_row = 1 if skip_first_row else 0
    choices: list[str] = []

    # BGR colors (OpenCV): yellow = scanned ring, green = selected
    bgr_scanned = (0, 255, 255)
    bgr_selected = (0, 255, 0)
    set_debug_bgr: np.ndarray | None = set_roi.copy() if debug_visual else None

    for i in range(start_row, rows):
        y_top, y_bot = row_edges[i]
        cell = set_bin[y_top:y_bot, :]
        if cell.size == 0:
            continue
        ch, cw = cell.shape[:2]
        cx, cy_cell = cw // 2, ch // 2
        radius = int(0.35 * min(cw, ch))
        if radius <= 1:
            continue
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(mask, (cx, cy_cell), radius, 255, -1)
        ink = int(np.sum(cell[mask == 255] > 0))
        tot = int(np.sum(mask == 255))
        ratio = ink / float(tot + 1e-6)
        py_roi = y_top + cy_cell
        px_roi = cx
        abs_x = x1 + px_roi
        abs_y = y1 + py_roi
        selected = ratio >= fill_threshold
        letter = chr(ord("a") + (i - start_row))

        if debug_print:
            sel_tag = "SELECTED" if selected else "not selected"
            print(
                f"[SET] row {i} ({letter})  ratio={ratio:.3f}  {sel_tag}  "
                f"(threshold={fill_threshold})"
            )

        if debug_visual and set_debug_bgr is not None:
            cv2.circle(set_debug_bgr, (px_roi, py_roi), radius, bgr_scanned, 1)
            if selected:
                cv2.circle(set_debug_bgr, (px_roi, py_roi), radius, bgr_selected, 2)

            cv2.circle(out_bgr, (abs_x, abs_y), radius, bgr_scanned, 1)
            if selected:
                cv2.circle(out_bgr, (abs_x, abs_y), radius, bgr_selected, 2)

        if selected:
            idx = i - start_row
            choices.append(chr(ord("a") + idx))

    set_label = "|".join(choices) if choices else ""

    # Zoom crop view for plotting (uses drawn overlay on out_bgr)
    pad = max(0, int(crop_pad))
    sy1 = max(0, y1 - pad)
    sy2 = min(out_bgr.shape[0], y2 + pad)
    sx1 = max(0, x1 - pad)
    sx2 = min(out_bgr.shape[1], x2 + pad)
    set_crop = out_bgr[sy1:sy2, sx1:sx2]
    set_crop_view = cv2.resize(set_crop, crop_view_size, interpolation=cv2.INTER_NEAREST)

    return SetCodeResult(
        set_roi=set_roi,
        set_bin=set_bin,
        set_label=set_label,
        set_crop_view=set_crop_view,
        set_debug_bgr=set_debug_bgr,
    )
