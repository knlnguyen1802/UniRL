"""Prefetch the PaddleOCR weights UniRL's OCR reward will load.

This does not install PaddleX extras. If ``paddlex.require_extra('ocr')`` still
fails, prefetch fails the same way — install the missing distributions first.

HuggingFace Xet downloads often stall or abort mid-file (CAS reconstruction).
Prefetch with Xet disabled, and optionally switch the PaddleX hoster:

    export HF_HUB_DISABLE_XET=1
    # optional: avoid huggingface entirely
    # export PADDLE_PDX_MODEL_SOURCE=bos   # or aistudio / modelscope
    # optional: put the cache on a shared disk
    # export PADDLE_PDX_CACHE_HOME=/data/n0090/UNI_RL/models/paddlex
    python benchmarks/speed_benchmarks/verl_omni/prefetch_ocr_models.py

Weights land in ``$PADDLE_PDX_CACHE_HOME/official_models`` (default
``~/.paddlex/official_models``). Point a later job at them with:

    export UNIRL_OCR_DET_DIR=$PADDLE_PDX_CACHE_HOME/official_models/<det-dir>
    export UNIRL_OCR_REC_DIR=$PADDLE_PDX_CACHE_HOME/official_models/<rec-dir>
"""

from __future__ import annotations

import os
from pathlib import Path

# huggingface_hub Xet/CAS is what aborted the last official-model fetch.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _missing_extras() -> list[str]:
    try:
        from paddlex.utils.deps import EXTRAS, is_dep_available
    except Exception:
        return ["paddlex"]
    missing: list[str] = []
    for extra in ("ocr-core", "ocr"):
        extra_deps = EXTRAS.get(extra) or {}
        missing.extend(dep for dep in extra_deps if not is_dep_available(dep))
    return sorted(set(missing))


def main() -> None:
    missing = _missing_extras()
    if missing:
        raise SystemExit(
            "PaddleX extras still missing; prefetch cannot construct PaddleOCR.\n"
            f"  pip install {' '.join(missing)}"
        )

    import paddle
    from paddleocr import PaddleOCR

    paddle.set_device("cpu")
    ocr_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "lang": os.environ.get("UNIRL_OCR_LANG", "en"),
        "device": "cpu",
    }
    det_dir = os.environ.get("UNIRL_OCR_DET_DIR", "").strip()
    rec_dir = os.environ.get("UNIRL_OCR_REC_DIR", "").strip()
    if det_dir:
        ocr_kwargs["text_detection_model_dir"] = det_dir
    if rec_dir:
        ocr_kwargs["text_recognition_model_dir"] = rec_dir
    ocr = PaddleOCR(**ocr_kwargs)
    cache_home = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
    cache = cache_home / "official_models"
    print(f"PaddleOCR constructed. Cache: {cache}")
    if cache.is_dir():
        for path in sorted(p for p in cache.iterdir() if p.is_dir()):
            print(f"  {path}")
        print("Export these before training to skip a re-download:")
        print(f"  export PADDLE_PDX_CACHE_HOME={cache_home}")
        print("  export UNIRL_OCR_DET_DIR=<det dir from the list above>")
        print("  export UNIRL_OCR_REC_DIR=<rec dir from the list above>")
    del ocr


if __name__ == "__main__":
    main()
