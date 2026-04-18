from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QuestionAnswersResult:
    answers_by_q: dict[int, list[str]]
    question_ink_ratio: float
    question_is_filled: bool
    q_bin: np.ndarray
    roi_edges_img: tuple[np.ndarray, np.ndarray]  # (y_edges, x_edges) in image space


def get_question_answers(
    *,
    resized_bgr: np.ndarray,
    out_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    q_start: int = 1,
    rows: int = 25,
    cols: int = 5,
    skip_first_col: bool = True,
    cell_fill_threshold: float = 0.22,
    cell_fill_threshold_inner: float | None = None,
    cell_fill_threshold_top_left: float | None = None,
    cell_fill_threshold_top_right: float | None = None,
    question_fill_threshold: float = 0.03,
    margin: int = 3,
    draw_detection_circles: bool = True,
) -> QuestionAnswersResult:
    """
    Question grid:
    - Draw purple ROI border
    - Draw red grid lines (rows x cols)
    - Detect filled answers per row (default: skip col0, cols 1..4 => a|b|c|d)
    - Mark filled cells with yellow circle
    """
    purple_bgr = (255, 0, 255)
    red_bgr = (0, 0, 255)
    yellow_bgr = (0, 255, 255)

    cv2.rectangle(out_bgr, (x1, y1), (x2, y2), purple_bgr, 2)

    y_edges_img = np.linspace(y1, y2, rows + 1, dtype=np.float32).astype(int)
    x_edges_img = np.linspace(x1, x2, cols + 1, dtype=np.float32).astype(int)
    y_edges_img[0], y_edges_img[-1] = y1, y2
    x_edges_img[0], x_edges_img[-1] = x1, x2

    for y in y_edges_img:
        cv2.line(out_bgr, (x1, int(y)), (x2, int(y)), red_bgr, 1)
    for x in x_edges_img:
        cv2.line(out_bgr, (int(x), y1), (int(x), y2), red_bgr, 1)

    q_roi = resized_bgr[y1 : y2 + 1, x1 : x2 + 1]
    q_gray = cv2.cvtColor(q_roi, cv2.COLOR_BGR2GRAY)
    q_blur = cv2.GaussianBlur(q_gray, (5, 5), 0)
    q_bin = cv2.adaptiveThreshold(
        q_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    q_bin = cv2.morphologyEx(q_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    if q_bin.shape[0] > 2 * margin and q_bin.shape[1] > 2 * margin:
        q_bin_inner = q_bin[margin:-margin, margin:-margin]
    else:
        q_bin_inner = q_bin

    question_ink_ratio = float(np.mean(q_bin_inner > 0)) if q_bin_inner.size else 0.0
    question_is_filled = question_ink_ratio >= float(question_fill_threshold)

    bin_h, bin_w = q_bin.shape[:2]
    y_edges_bin = np.linspace(0, bin_h, rows + 1, dtype=np.float32).astype(int)
    x_edges_bin = np.linspace(0, bin_w, cols + 1, dtype=np.float32).astype(int)
    y_edges_bin[0], y_edges_bin[-1] = 0, bin_h
    x_edges_bin[0], x_edges_bin[-1] = 0, bin_w

    answers_by_q: dict[int, list[str]] = {}
    start_c = 1 if skip_first_col else 0
    for r in range(rows):
        row_choices: list[str] = []
        by0, by1 = int(y_edges_bin[r]), int(y_edges_bin[r + 1])
        if by1 <= by0:
            answers_by_q[q_start + r] = []
            continue

        for c in range(start_c, cols):
            bx0, bx1 = int(x_edges_bin[c]), int(x_edges_bin[c + 1])
            if bx1 <= bx0:
                continue
            cell = q_bin[by0:by1, bx0:bx1]
            if cell.size == 0:
                continue

            ch, cw = cell.shape[:2]
            cx, cy = cw // 2, ch // 2
            radius_extra_px = 5
            radius = int(0.35 * min(cw, ch)) + radius_extra_px
            radius = min(radius, cx, cy, (cw - 1 - cx), (ch - 1 - cy))
            if radius <= 1:
                continue

            ix0, ix1 = int(x_edges_img[c]), int(x_edges_img[c + 1])
            iy0, iy1 = int(y_edges_img[r]), int(y_edges_img[r + 1])
            ccx = (ix0 + ix1) // 2
            ccy = (iy0 + iy1) // 2

            if draw_detection_circles:
                cv2.circle(out_bgr, (ccx, ccy), radius, (160, 160, 160), 1)

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

            thr_inner = (
                float(cell_fill_threshold_inner)
                if cell_fill_threshold_inner is not None
                else float(cell_fill_threshold) * 0.75
            )
            # Back-compat: if caller still passes top_right, treat it as top_left.
            effective_top_left = (
                cell_fill_threshold_top_left
                if cell_fill_threshold_top_left is not None
                else cell_fill_threshold_top_right
            )
            thr_tl = (
                float(effective_top_left)
                if effective_top_left is not None
                else float(cell_fill_threshold) * 0.60
            )

            if (ratio >= float(cell_fill_threshold)) or (ratio_inner >= thr_inner) or (ratio_tl >= thr_tl):
                label = chr(ord("a") + (c - start_c))
                row_choices.append(label)

                cv2.circle(out_bgr, (ccx, ccy), radius, yellow_bgr, 2)

        answers_by_q[q_start + r] = row_choices

    return QuestionAnswersResult(
        answers_by_q=answers_by_q,
        question_ink_ratio=question_ink_ratio,
        question_is_filled=question_is_filled,
        q_bin=q_bin,
        roi_edges_img=(y_edges_img, x_edges_img),
    )

