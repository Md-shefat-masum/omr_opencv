"""
HTTP API for OMR extraction. Keeps transport concerns here; extraction logic lives in app.py / app100.py.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Extraction entry points only — no OMR implementation in this module.
from app import exact_omr_result as extract_50_mark
from app100 import exact_omr_result as extract_100_mark

MAX_IMAGES = 50

app = FastAPI(title="OMR Extract API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _suffix_from_upload(upload: UploadFile) -> str:
    name = (upload.filename or "").strip()
    if name and "." in name:
        return Path(name).suffix
    ctype = (upload.content_type or "").lower()
    if "png" in ctype:
        return ".png"
    if "jpeg" in ctype or "jpg" in ctype:
        return ".jpg"
    if "bmp" in ctype:
        return ".bmp"
    return ".bin"


def _normalize_uploads(files: list[UploadFile] | None) -> list[UploadFile]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")
    return files[:MAX_IMAGES]


async def _run_batch(
    uploads: list[UploadFile],
    extract_fn: Callable[..., dict[str, Any]],
) -> JSONResponse:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="omr_api_") as tmp:
        tmp_path = Path(tmp)
        for idx, upload in enumerate(uploads):
            stem = Path(upload.filename or f"file_{idx}").stem or f"file_{idx}"
            in_path = tmp_path / f"{idx:03d}_{stem}{_suffix_from_upload(upload)}"
            out_path = tmp_path / f"{idx:03d}_{stem}_out.png"
            try:
                content = await upload.read()
                if not content:
                    results.append(
                        {
                            "original_filename": upload.filename,
                            "error": "Empty file",
                        }
                    )
                    continue
                in_path.write_bytes(content)
                data = extract_fn(str(in_path), str(out_path), show=False)
                results.append(
                    {
                        "original_filename": upload.filename,
                        "result": _json_safe(data),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "original_filename": upload.filename,
                        "error": str(e),
                    }
                )
    return JSONResponse(content={"count": len(results), "results": results})


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "on", "service": "omr-extract"}


@app.post("/50-mark-extract")
async def mark_extract_50(
    files: Annotated[
        list[UploadFile] | None,
        File(description="One or more images (max 50 used); field name: files"),
    ] = None,
) -> JSONResponse:
    uploads = _normalize_uploads(files)
    return await _run_batch(uploads, extract_50_mark)


@app.post("/100-mark-extract")
async def mark_extract_100(
    files: Annotated[
        list[UploadFile] | None,
        File(description="One or more images (max 50 used); field name: files"),
    ] = None,
) -> JSONResponse:
    uploads = _normalize_uploads(files)
    return await _run_batch(uploads, extract_100_mark)
