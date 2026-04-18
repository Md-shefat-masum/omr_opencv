"""
Refine approximate corner markers (tl, tr, bl, br) to the actual square vertices.

Strategy per corner:
1. ROI around the initial point; **bottom-right** uses a SE-biased window (narrow left margin,
   nudged toward the sheet corner) so a nearby filled bubble is not inside the ROI; optional
   stronger opening breaks a thin bridge between bubble and square.
2. **Bottom-right** additionally: scan candidates with diagonal priority (outer vertex closest to
   the image corner, tie-breaking toward the bottom-right), validate each minAreaRect with
   per-corner ink / inward-darkness checks, and accept when at least three corners look square-like
   (or all four). If that path finds nothing, fall back to the generic picker below.
3. Pick the square-like contour whose **outer vertex is closest to the sheet corner** (not
   “largest area”), so a merged blob / bubble does not win over the actual marker.
4. Final point: convex-hull vertex nearest the sheet corner for tl/tr/bl; **br** uses the
   min-area-rectangle outer vertex so edge-clipped markers still aim at the true sheet corner.
5. Refine with guarded cornerSubPix; Sobel fallback if no contour (non-BR corners).

**Green frame geometry (default ``br``):** The debug overlay draws the bottom edge ``bl→br`` and
right edge ``tr→br``. Those are the same two lines as completing a parallelogram from ``tl, tr, bl``
(opposite sides parallel). The intersection is ``br = tr + bl - tl``. Using that for ``br`` aligns
the frame corner with the edges implied by the three good corners. For screenshots that already
contain green lines, see ``detect_green_overlay_corner_bgr``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    from utils.corner_markers import _is_corner_square_candidate, _outermost_corner_for_marker
except ImportError:
    _parent = Path(__file__).resolve().parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from utils.corner_markers import _is_corner_square_candidate, _outermost_corner_for_marker


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _fallback_edge_snap(
    gray: np.ndarray,
    mag: np.ndarray,
    initial_xy: tuple[float, float],
    win: int,
) -> np.ndarray:
    """Snap toward strong edges in a window (improves on raw max-Canny pixel)."""
    h, w = gray.shape[:2]
    ix, iy = int(round(initial_xy[0])), int(round(initial_xy[1]))
    best = np.array([ix, iy], dtype=np.float32)
    best_score = -1.0
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            nx, ny = ix + dx, iy + dy
            if nx < 1 or ny < 1 or nx >= w - 1 or ny >= h - 1:
                continue
            s = float(mag[ny, nx])
            if s > best_score:
                best_score = s
                best = np.array([nx, ny], dtype=np.float32)
    return best


def br_corner_from_parallelogram_closure(
    tl: np.ndarray | tuple[float, float],
    tr: np.ndarray | tuple[float, float],
    bl: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """
    Fourth corner of the parallelogram defined by ``tl``, ``tr``, ``bl``.

    This is where the **bottom** edge (parallel to ``tl→tr`` through ``bl``) meets the **right**
    edge (parallel to ``tl→bl`` through ``tr``) — the same corner as the green ``L`` in the
    overlay (intersection of the bottom and right frame lines).
    """
    t = np.asarray(tl, dtype=np.float64).reshape(2)
    r = np.asarray(tr, dtype=np.float64).reshape(2)
    b = np.asarray(bl, dtype=np.float64).reshape(2)
    return (r + b - t).astype(np.float32)


def _clip_xy_to_image(pt: np.ndarray, w: int, h: int) -> np.ndarray:
    out = pt.astype(np.float32).reshape(2).copy()
    out[0] = float(np.clip(out[0], 0.0, float(w - 1)))
    out[1] = float(np.clip(out[1], 0.0, float(h - 1)))
    return out


def detect_green_overlay_corner_bgr(
    bgr: np.ndarray,
    *,
    bottom_right_roi_frac: tuple[float, float] = (0.55, 0.55),
) -> tuple[float, float] | None:
    """
    Locate the bottom-right **L** corner of bright green overlay lines (debug screenshots).

    Uses HSV green thresholding on a bottom-right ROI, then takes the column with strongest
    green mass (vertical leg) and the row with strongest green mass (horizontal leg); their
    intersection approximates the inner corner of the ``L``. Returns ``None`` if no green found.

    Parameters
    ----------
    bottom_right_roi_frac
        Fraction ``(fw, fh)`` of width and height to keep from the bottom-right of the image.
    """
    h, w = bgr.shape[:2]
    fw, fh = bottom_right_roi_frac
    x0 = max(0, int(w * (1.0 - fw)))
    y0 = max(0, int(h * (1.0 - fh)))
    roi = bgr[y0:h, x0:w]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Bright green overlay (OpenCV H 0–179)
    lower = np.array([40, 60, 60], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if not np.any(mask):
        return None

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)
    rel_x = int(np.argmax(col_sums))
    rel_y = int(np.argmax(row_sums))
    return float(x0 + rel_x), float(y0 + rel_y)


def _roi_bounds_for_corner(
    key: str,
    xi: int,
    yi: int,
    w: int,
    h: int,
    search_radius: int,
) -> tuple[int, int, int, int]:
    """
    Default: symmetric ROI. For ``br``, bias toward the sheet bottom-right so the search
    sits on the corner square, not on a filled bubble immediately to the left (e.g. Q50).
    """
    sr = search_radius
    if key == "br":
        # Nudge anchor forward (SE): user / layout: square is slightly past the initial seed.
        xi_a = min(w - 1, xi + 22)
        yi_a = min(h - 1, yi + 10)
        left_keep = max(16, sr // 6)
        up_keep = max(28, int(sr * 0.55))
        x0 = max(0, xi_a - left_keep)
        y0 = max(0, yi_a - up_keep)
        x1 = min(w, xi_a + sr)
        y1 = min(h, yi_a + sr)
        return x0, y0, x1, y1
    x0 = max(0, xi - sr)
    y0 = max(0, yi - sr)
    x1 = min(w, xi + sr)
    y1 = min(h, yi + sr)
    return x0, y0, x1, y1


def _outer_corner_from_contour_hull(
    cnt: np.ndarray,
    x_off: float,
    y_off: float,
    outer_ref: tuple[float, float],
) -> np.ndarray:
    """Vertex of the contour convex hull closest to the sheet corner (outer corner of ink blob)."""
    c = cnt.reshape(-1, 2).astype(np.float32)
    c[:, 0] += x_off
    c[:, 1] += y_off
    if len(c) < 3:
        p = c[0] if len(c) else np.array(outer_ref, dtype=np.float32)
        return p
    hull = cv2.convexHull(c.astype(np.float32))
    pts = hull.reshape(-1, 2).astype(np.float32)
    ref = np.array(outer_ref, dtype=np.float32)
    d = np.sum((pts - ref) ** 2, axis=1)
    return pts[int(np.argmin(d))]


def _outer_corner_from_min_area_rect(
    cnt: np.ndarray,
    x_off: float,
    y_off: float,
    outer_ref: tuple[float, float],
) -> np.ndarray:
    """
    Outer sheet-corner vertex of the fitted min-area rectangle (full-image coords).

    For **bottom-right** markers flush with the image edge, the convex hull of the visible ink
    often lacks the true SE corner; the hull vertex nearest the sheet corner can sit too far
    up-left (wrong corner of the blob). minAreaRect extrapolates the square so the vertex chosen
    by ``_outermost_corner_for_marker`` tracks the real registration corner.
    """
    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    if rw <= 0 or rh <= 0:
        return np.array(outer_ref, dtype=np.float32)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += x_off
    box[:, 1] += y_off
    return _outermost_corner_for_marker(box, outer_ref).astype(np.float32)


# Small empirical offset after BR refine (+x = right, +y = down). Reduced once BR uses minAreaRect.
BR_NUDGE_DX = 2.0
BR_NUDGE_DY = 5.0

# Bottom-right: accept a candidate if at least this many box corners look square-like.
BR_MIN_SQUARE_CORNERS = 3


def _vertex_square_like_at_corner(
    gray: np.ndarray,
    thr: np.ndarray,
    w: int,
    h: int,
    vx: float,
    vy: float,
    cx: float,
    cy: float,
) -> bool:
    """
    Heuristic: a corner of a filled square has ink nearby and darker pixels inward toward the centroid.
    """
    xi = int(round(vx))
    yi = int(round(vy))
    r = 5
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    patch = thr[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    ink_ratio = float(np.mean(patch > 127))

    dx, dy = cx - vx, cy - vy
    d = float(np.hypot(dx, dy))
    if d < 2.0:
        return ink_ratio > 0.28
    ux, uy = dx / d, dy / d
    samples: list[float] = []
    for t in (2.5, 5.0, 7.5, 10.0):
        sx = int(round(vx + ux * t))
        sy = int(round(vy + uy * t))
        if 0 <= sx < w and 0 <= sy < h:
            samples.append(float(gray[sy, sx]))
    inward_dark = bool(samples) and float(np.mean(samples)) < 168.0

    if ink_ratio > 0.32:
        return True
    if inward_dark and ink_ratio > 0.07:
        return True
    return False


def _count_square_like_box_corners(
    gray: np.ndarray,
    thr: np.ndarray,
    box: np.ndarray,
    cx_full: float,
    cy_full: float,
    w: int,
    h: int,
) -> int:
    """How many minAreaRect vertices pass local + inward ink checks (0..4)."""
    n = 0
    for i in range(4):
        vx, vy = float(box[i, 0]), float(box[i, 1])
        if _vertex_square_like_at_corner(gray, thr, w, h, vx, vy, cx_full, cy_full):
            n += 1
    return n


def _br_diagonal_priority_score(
    outer_ref: tuple[float, float],
    outer_vertex: np.ndarray,
    centroid: np.ndarray,
    initial_xy: tuple[float, float],
) -> tuple[float, float, float]:
    """
    Lower is better: prefer markers whose outer corner sits closest to the image BR (scan from the
    sheet corner inward), then prefer larger (x+y) on the outer vertex (more toward SE), then
    centroid near the initial seed.
    """
    ox, oy = float(outer_ref[0]), float(outer_ref[1])
    vx, vy = float(outer_vertex[0]), float(outer_vertex[1])
    cx, cy = float(centroid[0]), float(centroid[1])
    iix, iiy = float(initial_xy[0]), float(initial_xy[1])
    corner_d2 = (vx - ox) ** 2 + (vy - oy) ** 2
    # Tie-break: outer vertex with larger (x+y) is closer to bottom-right of the image.
    centroid_d2 = (cx - iix) ** 2 + (cy - iiy) ** 2
    return (corner_d2, -(vx + vy), centroid_d2)


def _pick_br_contour_diagonal_corner_validate(
    gray: np.ndarray,
    thr: np.ndarray,
    contours: list,
    x0: int,
    y0: int,
    w: int,
    h: int,
    outer_ref: tuple[float, float],
    initial_xy: tuple[float, float],
    ix: float,
    iy: float,
    max_r2: float,
    min_area: float,
) -> np.ndarray | None:
    """
    BR-only: evaluate square candidates with 4-corner checks; accept if >= BR_MIN_SQUARE_CORNERS.
    Order by corner-match count (4 before 3), then diagonal priority (outer vertex closest to
    image BR, i.e. scanning from the sheet corner toward the top-left).
    Centroid near the initial seed is preferred as a tie-break but not required (looser than the
    generic picker so a bad seed does not miss the real marker).
    """
    ix_f, iy_f = float(ix), float(iy)
    max_centroid_d2 = max_r2 * (2.2**2)

    def _collect(require_centroid: bool) -> list[tuple[int, tuple, np.ndarray]]:
        scored: list[tuple[int, tuple, np.ndarray]] = []
        for cnt in contours:
            ok, feats = _is_corner_square_candidate(cnt)
            if not ok:
                continue
            area = feats["area"]
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (ccx, ccy), (bw, bh), _ = rect
            if bw <= 0 or bh <= 0:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 0] += x0
            box[:, 1] += y0
            cx_full = ccx + x0
            cy_full = ccy + y0
            c_dist2 = (cx_full - ix_f) ** 2 + (cy_full - iy_f) ** 2
            if require_centroid and c_dist2 > max_centroid_d2:
                continue

            n_match = _count_square_like_box_corners(gray, thr, box, cx_full, cy_full, w, h)
            v = _outermost_corner_for_marker(box, outer_ref).astype(np.float32)
            centroid = np.array([cx_full, cy_full], dtype=np.float32)
            pri = _br_diagonal_priority_score(outer_ref, v, centroid, initial_xy)
            scored.append((n_match, pri, cnt))

        return scored

    scored = _collect(require_centroid=True)
    if not scored:
        scored = _collect(require_centroid=False)

    if not scored:
        return None

    scored.sort(key=lambda t: (-t[0], t[1][0], t[1][1], t[1][2]))

    for n_match, _pri, cnt in scored:
        if n_match >= BR_MIN_SQUARE_CORNERS:
            return cnt
    return None


def _maybe_br_corner_nudge(pt: np.ndarray, key: str, w: int, h: int) -> np.ndarray:
    if key != "br":
        return pt
    out = pt.astype(np.float32).reshape(2).copy()
    out[0] += BR_NUDGE_DX
    out[1] += BR_NUDGE_DY
    out[0] = float(np.clip(out[0], 0.0, float(w - 1)))
    out[1] = float(np.clip(out[1], 0.0, float(h - 1)))
    return out


def _sheet_corner_ref(key: str, w: int, h: int) -> tuple[float, float]:
    """Image-corner reference used to pick the *outer* vertex of the marker (same idea as corner_markers)."""
    wm, hm = float(w - 1), float(h - 1)
    if key == "tl":
        return (0.0, 0.0)
    if key == "tr":
        return (wm, 0.0)
    if key == "bl":
        return (0.0, hm)
    if key == "br":
        return (wm, hm)
    raise ValueError(f"unknown corner key: {key!r}")


def _local_median_gray(gray: np.ndarray, x: float, y: float, r: int = 4) -> float:
    h, w = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 255.0
    return float(np.median(patch))


def _refine_subpix(
    gray: np.ndarray,
    pt: np.ndarray,
    *,
    max_drift_px: float = 6.0,
    win: int = 7,
    max_luma_delta: float = 18.0,
) -> np.ndarray:
    """
    Subpixel refinement. Rejects shifts that land on paler paper (common BR failure:
    cornerSubPix follows a strong edge just outside the ink).
    """
    orig = pt.astype(np.float32).reshape(2)
    m0 = _local_median_gray(gray, float(orig[0]), float(orig[1]))
    pt_in = orig.reshape(1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.001)
    refined = cv2.cornerSubPix(gray, pt_in, (win, win), (-1, -1), criteria)
    out = refined.reshape(2).astype(np.float32)
    if np.linalg.norm(out - orig) > max_drift_px:
        return orig
    m1 = _local_median_gray(gray, float(out[0]), float(out[1]))
    # Paler patch after refine ⇒ likely slid onto white / wrong edge — keep minAreaRect vertex.
    if m1 > m0 + max_luma_delta:
        return orig
    return out


def refine_corner_from_initial(
    gray: np.ndarray,
    key: str,
    initial_xy: tuple[float, float],
    *,
    search_radius: int = 95,
    mag: np.ndarray | None = None,
) -> np.ndarray:
    """
    Find the actual outer corner of the black square marker near ``initial_xy``.

    ROI placement and contour choice use ``initial_xy``. Which vertex of the fitted
    rectangle is the *outer* corner uses the sheet edge (tl→(0,0), tr→(w-1,0), …),
    matching ``utils.corner_markers`` so a biased initial point cannot pick the
    wrong vertex of the rotated rect.
    """
    h, w = gray.shape[:2]
    outer_ref = _sheet_corner_ref(key, w, h)
    ix, iy = float(initial_xy[0]), float(initial_xy[1])
    xi, yi = int(round(ix)), int(round(iy))

    x0, y0, x1, y1 = _roi_bounds_for_corner(key, xi, yi, w, h, search_radius)
    rw, rh = x1 - x0, y1 - y0
    if rw < 16 or rh < 16:
        m = mag if mag is not None else _sobel_mag(gray)
        return _maybe_br_corner_nudge(
            _fallback_edge_snap(gray, m, initial_xy, win=min(25, search_radius // 2)),
            key,
            w,
            h,
        )

    roi = gray[y0:y1, x0:x1]
    thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k3, iterations=1)
    if key == "br":
        # Break a thin link between a nearby filled bubble and the corner square.
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k5, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_ref_arr = np.array(outer_ref, dtype=np.float32)
    max_r2 = (float(search_radius) * 0.82) ** 2
    min_area = 32.0

    if key == "br":
        br_pick = _pick_br_contour_diagonal_corner_validate(
            gray,
            thr,
            contours,
            x0,
            y0,
            w,
            h,
            outer_ref,
            initial_xy,
            ix,
            iy,
            max_r2,
            min_area,
        )
        if br_pick is not None:
            corner = _outer_corner_from_min_area_rect(
                br_pick, float(x0), float(y0), outer_ref
            )
            return _maybe_br_corner_nudge(corner, key, w, h)

    def _pick_contour(require_centroid: bool) -> tuple[float, float, float, np.ndarray] | None:
        best_local: tuple[float, float, float, np.ndarray] | None = None
        for cnt in contours:
            ok, feats = _is_corner_square_candidate(cnt)
            if not ok:
                continue
            area = feats["area"]
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (ccx, ccy), (bw, bh), _ = rect
            if bw <= 0 or bh <= 0:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 0] += x0
            box[:, 1] += y0
            cx_full = ccx + x0
            cy_full = ccy + y0
            v = _outermost_corner_for_marker(box, outer_ref).astype(np.float32)
            corner_d2 = float(np.sum((v - outer_ref_arr) ** 2))
            c_dist2 = (cx_full - ix) ** 2 + (cy_full - iy) ** 2

            if require_centroid and c_dist2 > max_r2 * 1.15:
                continue

            cand = (corner_d2, -area, c_dist2, cnt)
            if best_local is None:
                best_local = cand
                continue
            if cand[0] < best_local[0] - 1e-6:
                best_local = cand
            elif abs(cand[0] - best_local[0]) <= 1e-6:
                if cand[1] < best_local[1] or (
                    cand[1] == best_local[1] and cand[2] < best_local[2]
                ):
                    best_local = cand
        return best_local

    # Prefer contour whose minAreaRect outer vertex is closest to the sheet corner (square beats bubble).
    best = _pick_contour(require_centroid=True)
    if best is None:
        best = _pick_contour(require_centroid=False)

    if best is None:
        m = mag if mag is not None else _sobel_mag(gray)
        return _maybe_br_corner_nudge(
            _fallback_edge_snap(gray, m, initial_xy, win=min(30, search_radius // 2)),
            key,
            w,
            h,
        )

    best_cnt = best[3]
    if key == "br":
        corner = _outer_corner_from_min_area_rect(best_cnt, float(x0), float(y0), outer_ref)
        return _maybe_br_corner_nudge(corner, key, w, h)
    corner = _outer_corner_from_contour_hull(best_cnt, float(x0), float(y0), outer_ref).astype(np.float32)
    return _maybe_br_corner_nudge(_refine_subpix(gray, corner), key, w, h)


def refine_four_corners(
    gray: np.ndarray,
    points: dict[str, tuple[float, float]],
    search_radius: int = 95,
    *,
    br_mode: Literal["refined", "parallelogram", "blend"] = "parallelogram",
    br_blend: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Refine all four named corners; computes Sobel magnitude once for fallbacks.

    Parameters
    ----------
    br_mode
        ``parallelogram`` (default): set ``br`` to the intersection of the bottom line through
        ``bl`` (parallel to top) and the right line through ``tr`` (parallel to left), i.e.
        ``tr + bl - tl``, so the green overlay corner matches the frame implied by the other corners.
        ``refined``: keep contour-refined ``br``. ``blend``: convex mix of both (see ``br_blend``).
    br_blend
        When ``br_mode == "blend"``: weight on the parallelogram corner; ``1 - br_blend`` on refined.
    """
    h, w = gray.shape[:2]
    mag = _sobel_mag(gray)
    refined: dict[str, np.ndarray] = {
        k: refine_corner_from_initial(
            gray,
            k,
            (float(x), float(y)),
            search_radius=search_radius,
            mag=mag,
        )
        for k, (x, y) in points.items()
    }

    if br_mode == "refined":
        return refined

    br_para = br_corner_from_parallelogram_closure(
        refined["tl"], refined["tr"], refined["bl"]
    )
    br_para = _clip_xy_to_image(br_para, w, h)

    if br_mode == "parallelogram":
        refined["br"] = br_para
        return refined

    # blend
    a = float(np.clip(br_blend, 0.0, 1.0))
    refined["br"] = ((1.0 - a) * refined["br"].astype(np.float64) + a * br_para.astype(np.float64)).astype(
        np.float32
    )
    refined["br"] = _clip_xy_to_image(refined["br"], w, h)
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine OMR sheet corner markers from approximate points.")
    parser.add_argument("image", nargs="?", default="6.jpg", help="Input image path")
    parser.add_argument("--width", type=int, default=550, help="Resize width")
    parser.add_argument("--height", type=int, default=820, help="Resize height")
    parser.add_argument("--search-radius", type=int, default=95, help="ROI half-size around each initial point")
    parser.add_argument(
        "--br-mode",
        choices=("refined", "parallelogram", "blend"),
        default="parallelogram",
        help="How to set bottom-right: parallelogram closure from tl,tr,bl (default), refined contour, or blend",
    )
    parser.add_argument(
        "--br-blend",
        type=float,
        default=0.5,
        help="When --br-mode=blend, weight on parallelogram br (0=refined only, 1=parallelogram only)",
    )
    parser.add_argument(
        "--print-green-corner",
        action="store_true",
        help="Detect bright green overlay L-corner in BGR (debug screenshots) and print coordinates",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI windows")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path.resolve()}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")
    img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)

    if args.print_green_corner:
        gc = detect_green_overlay_corner_bgr(img)
        if gc is None:
            print("Green overlay corner: not found (no green pixels in bottom-right ROI)")
        else:
            print(f"Green overlay L-corner (approx): ({gc[0]:.2f}, {gc[1]:.2f})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape

    # Approximate corners (adjust to your sheet / detector output)
    points: dict[str, tuple[float, float]] = {
        "tl": (50, 50),
        "tr": (508, 50),
        # "br": (508, 775),
        "br": (515, 785),
        "bl": (50, 775),
    }

    refined_float = refine_four_corners(
        gray,
        points,
        search_radius=args.search_radius,
        br_mode=args.br_mode,
        br_blend=args.br_blend,
    )
    refined = {k: (int(round(v[0])), int(round(v[1]))) for k, v in refined_float.items()}

    print("=== REFINED CORNERS (float xy, then integer) ===")
    for k in ("tl", "tr", "br", "bl"):
        rf = refined_float[k]
        ri = refined[k]
        print(f"{k}: ({rf[0]:.3f}, {rf[1]:.3f}) -> {ri}")

    # --- visualization ---
    gray_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    final_vis = img.copy()

    # for p in points.values():
    #     cv2.circle(gray_vis, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
    #     cv2.circle(edge_vis, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
    #     cv2.circle(final_vis, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

    # for k, rf in refined_float.items():
    #     p = (int(round(rf[0])), int(round(rf[1])))
    #     cv2.circle(gray_vis, p, 5, (0, 0, 255), -1)
    #     cv2.circle(edge_vis, p, 5, (0, 0, 255), -1)
    #     cv2.circle(final_vis, p, 5, (0, 0, 255), -1)
    #     cv2.putText(
    #         final_vis,
    #         k,
    #         (p[0] + 6, p[1] - 6),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.45,
    #         (0, 255, 255),
    #         1,
    #         cv2.LINE_AA,
    #     )

    poly = np.array(
        [
            refined_float["tl"],
            refined_float["tr"],
            refined_float["br"],
            refined_float["bl"],
        ],
        dtype=np.float32,
    )
    cv2.polylines(final_vis, [np.round(poly).astype(np.int32)], True, (0, 255, 0), 2)

    if args.no_show:
        out_path = img_path.with_name(img_path.stem + "_corners_refined.jpg")
        cv2.imwrite(str(out_path), final_vis)
        print(f"Wrote: {out_path}")
        return

    cv2.imshow("1 - Grayscale View", gray_vis)
    cv2.imshow("2 - Edge View", edge_vis)
    cv2.imshow("3 - Final Border View", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
