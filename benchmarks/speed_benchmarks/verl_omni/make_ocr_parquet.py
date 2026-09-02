"""Convert this repo's ``datasets/ocr/{train,test}.txt`` into verl-omni parquet.

Schema matches ``verl-omni/examples/flowgrpo_trainer/data_process/qwenimage_ocr.py``
so the aligned Qwen-Image+FlowGRPO pair can share the same OCR prompt files.
Ground truth is the quoted span (``The image displays "..."``); the aligned
verl-omni launcher strips the system turn via ``custom_chat_template`` so both
sides encode the raw user line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "datasets" / "ocr"

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)
NEGATIVE_USER_PROMPT = " "


def extract_ground_truth(prompt: str) -> str:
    parts = prompt.split('"')
    if len(parts) < 3 or not parts[1].strip():
        raise ValueError(f"OCR prompt must contain quoted target text, got: {prompt[:120]!r}")
    return parts[1]


def build(split: str, filename: str) -> datasets.Dataset:
    prompts = [line.strip() for line in (SRC / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
    return datasets.Dataset.from_list(
        [
            {
                "data_source": "unirl/ocr",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": NEGATIVE_USER_PROMPT},
                ],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": extract_ground_truth(p)},
                "extra_info": {"split": split, "index": i, "raw_prompt": p},
            }
            for i, p in enumerate(prompts)
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/ocr/qwen_image")
    out = Path(ap.parse_args().out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train, test = build("train", "train.txt"), build("test", "test.txt")
    train.to_parquet(str(out / "train.parquet"))
    test.to_parquet(str(out / "test.parquet"))
    print(f"wrote {len(train)} train / {len(test)} test to {out}")
