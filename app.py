from __future__ import annotations

import json
import time
from pathlib import Path

from utils.get_coaching_code import get_coaching_code
from utils.get_question_answers import get_question_answers
from utils.get_set_code import get_set_code
from utils.omr_warp import (
    OMR_CANVAS_SIZE_WH,
    OMR_TEMPLATE_CORNERS,
    warp_resized_scan_to_template,
)
import cv2
import numpy as np

def exact_omr_result(
    input_path: str,
    output_path: str,
    *,
    target_size_wh: tuple[int, int] = (550, 820),
    show: bool = True,
) -> dict:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    if target_size_wh != OMR_CANVAS_SIZE_WH:
        raise ValueError(
            f"target_size_wh must be {OMR_CANVAS_SIZE_WH} for template warp; got {target_size_wh}"
        )

    resized = cv2.resize(img, target_size_wh, interpolation=cv2.INTER_AREA)

    # Perspective-normalize: map detected anchor corners → fixed template quad (see utils/omr_warp.py).
    normalized, refined_src_corners, perspective_m = warp_resized_scan_to_template(resized)
    resized = normalized

    form_square = np.array(
        [
            OMR_TEMPLATE_CORNERS["tl"],
            OMR_TEMPLATE_CORNERS["tr"],
            OMR_TEMPLATE_CORNERS["br"],
            OMR_TEMPLATE_CORNERS["bl"],
        ],
        dtype=np.int32,
    )
    out = resized.copy()
    cv2.polylines(out, [form_square], isClosed=True, color=(0, 255, 0), thickness=4)

    # Coaching code block
    x_min, y_min = 55, 75
    x_max, y_max = 220, 375
    coaching = get_coaching_code(
        resized_bgr=resized,
        out_bgr=out,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        cols=6,
        rows=11,
        grid_color_bgr=(255, 0, 0),
    )

    # SET code block
    set_result = get_set_code(
        resized_bgr=resized,
        out_bgr=out,
        x1=227,
        y1=75,
        x2=252,
        y2=238,
        rows=6,
        skip_first_row=True,
    )

    # Questions block
    q_x1, q_y1 = 245, 70
    q_x2, q_y2 = 375, 768
    q_rows, q_cols = 25, 5
    q_res_1 = get_question_answers(
        resized_bgr=resized,
        out_bgr=out,
        x1=q_x1,
        y1=q_y1,
        x2=q_x2,
        y2=q_y2,
        q_start=1,
        rows=q_rows,
        cols=q_cols,
        skip_first_col=True,
    )

    q_width = q_x2 - q_x1
    q2_x1 = q_x2 - 4
    q2_x2 = q2_x1 + q_width - 3
    q_res_2 = get_question_answers(
        resized_bgr=resized,
        out_bgr=out,
        x1=q2_x1,
        y1=q_y1,
        x2=q2_x2,
        y2=q_y2,
        q_start=26,
        rows=q_rows,
        cols=q_cols,
        skip_first_col=True,
    )

    filled_answers_by_q = {**q_res_1.answers_by_q, **q_res_2.answers_by_q}

    import os
    outputs_50_dir = 'outputs_50'
    os.makedirs(outputs_50_dir, exist_ok=True)
    output_path_merged = os.path.join(outputs_50_dir, os.path.basename(output_path))
    cv2.imwrite(output_path_merged, out)

    if show:
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        coaching_crop = out[y_min : y_max + 1, x_min : x_max + 1]

        fig = plt.figure(figsize=(12, 14))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2.2, 1], hspace=0.15, wspace=0.08)
        ax_out = fig.add_subplot(gs[0, :])
        ax_coach = fig.add_subplot(gs[1, 0])
        ax_set = fig.add_subplot(gs[1, 1])

        ax_out.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax_out.set_title("Result (perspective-normalized to template)")
        ax_out.axis("off")

        coaching_crop_view = cv2.resize(coaching_crop, (300, 400), interpolation=cv2.INTER_AREA)
        ax_coach.imshow(cv2.cvtColor(coaching_crop_view, cv2.COLOR_BGR2RGB))
        ax_coach.set_title("Coaching square (crop)")
        ax_coach.axis("off")

        # axes[1, 0].imshow(coaching.first_row_bin, cmap="gray")
        # axes[1, 0].set_title("First row (binary)")
        # axes[1, 0].axis("off")

        # axes[1, 1].imshow(coaching.first_row_edges, cmap="gray")
        # axes[1, 1].set_title(f"First row (edges) | filled_numbers={coaching.filled_numbers}")
        # axes[1, 1].axis("off")

        ax_set.imshow(cv2.cvtColor(set_result.set_crop_view, cv2.COLOR_BGR2RGB))
        ax_set.set_title(f"SET box (zoom) | set_label={set_result.set_label}")
        ax_set.axis("off")

        fig.tight_layout()
        plt.show()

        # q_crop = out[
        #     max(0, q_y1 - 8) : min(out.shape[0], q_y2 + 8),
        #     max(0, q_x1 - 8) : min(out.shape[1], q_x2 + 8),
        # ]
        # q_crop_view = cv2.resize(q_crop, (200, 500), interpolation=cv2.INTER_NEAREST)
        # q2_crop = out[
        #     max(0, q_y1 - 8) : min(out.shape[0], q_y2 + 8),
        #     max(0, q2_x1 - 8) : min(out.shape[1], q2_x2 + 8),
        # ]
        # q2_crop_view = cv2.resize(q2_crop, (200, 500), interpolation=cv2.INTER_NEAREST)

        # fig_q, axes_q = plt.subplots(1, 2, figsize=(16, 8))
        # ax_q1, ax_q2 = axes_q

        # ax_q1.imshow(cv2.cvtColor(q_crop_view, cv2.COLOR_BGR2RGB))
        # ax_q1.set_title(f"Questions Q1..Q25 | ink={q_res_1.question_ink_ratio:.3f}")
        # answers_text = "\n".join(
        #     f"{q:>2}: {''.join(filled_answers_by_q.get(q, [])) if filled_answers_by_q.get(q, []) else '-'}"
        #     for q in range(1, 26)
        # )
        # ax_q1.text(
        #     -0.02,
        #     0.5,
        #     answers_text,
        #     transform=ax_q1.transAxes,
        #     va="center",
        #     ha="right",
        #     fontsize=8,
        #     family="monospace",
        #     clip_on=False,
        # )
        # ax_q1.axis("off")

        # ax_q2.imshow(cv2.cvtColor(q2_crop_view, cv2.COLOR_BGR2RGB))
        # ax_q2.set_title("Questions Q26..Q50")
        # answers_text_2 = "\n".join(
        #     f"{q:>2}: {''.join(filled_answers_by_q.get(q, [])) if filled_answers_by_q.get(q, []) else '-'}"
        #     for q in range(26, 51)
        # )
        # ax_q2.text(
        #     -0.02,
        #     0.5,
        #     answers_text_2,
        #     transform=ax_q2.transAxes,
        #     va="center",
        #     ha="right",
        #     fontsize=8,
        #     family="monospace",
        #     clip_on=False,
        # )
        # ax_q2.axis("off")

        # plt.tight_layout()
        # plt.show()

    return {
        "coaching_matrix": coaching.filled_by_col,
        "coaching_codes": coaching.filled_numbers,
        "set_label": set_result.set_label,
        "answers_by_q": filled_answers_by_q,
        # "question_ink_ratio": q_res_1.question_ink_ratio,
        # "question_is_filled": q_res_1.question_is_filled,
        # "output_path": output_path,
        # "omr_canvas_size_wh": OMR_CANVAS_SIZE_WH,
        # "refined_src_corners": {k: v.tolist() for k, v in refined_src_corners.items()},
        # "perspective_matrix": perspective_m.tolist(),
    }


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    # res1 = exact_omr_result("1.jpg", "res.png", show=False)
    # res2 = exact_omr_result("2.jpg", "res2.png")
    # res3 = exact_omr_result("3.jpg", "res3.png")
    # res4 = exact_omr_result("4.jpg", "res4.png")
    # res5 = exact_omr_result("5.jpg", "res5.png")
    # res6 = exact_omr_result("7.bmp", "res6.png")
    # print(res1)
    # print(res2)
    # print(res3)
    # print(res4)
    # print(res5)
    # print(res6)

    omr_dir = Path("omr_images/50")
    bmp_paths = sorted(omr_dir.glob("*.bmp"))
    n_paths = len(bmp_paths)
    results: list[dict] = []
    t_total_start = time.perf_counter()
    for bmp_path in bmp_paths:
        out_path = omr_dir / f"res{bmp_path.stem}.png"
        t0 = time.perf_counter()
        result = exact_omr_result(str(bmp_path), str(out_path), show=False)
        elapsed_s = time.perf_counter() - t0
        entry = {
            "input_path": str(bmp_path),
            "output_path": str(out_path),
            "seconds": elapsed_s,
            **result,
        }
        results.append(entry)
        print(f"{bmp_path.name}: {elapsed_s:.4f}s")
    total_s = time.perf_counter() - t_total_start
    out_json = Path(__file__).resolve().parent / "50_q_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"paths: {n_paths} | total time: {total_s:.4f}s | wrote {out_json}")

