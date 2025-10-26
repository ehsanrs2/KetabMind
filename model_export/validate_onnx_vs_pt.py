"""Validate ONNX vs PyTorch logits for HooshvareLab/gpt2-fa."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TORCH_MODEL_ID = "HooshvareLab/gpt2-fa"
ONNX_DIR = Path(__file__).resolve().parent / "onnx" / "gpt2fa"
MAX_DIFF_THRESHOLD = 1e-3

PROMPTS = {
    "short": "سلام!",
    "medium": "سلام! حال شما چطور است؟ امیدوارم روز خوبی داشته باشید.",
    "long": (
        "سلام! امروز می‌خواستم دربارهٔ کتاب‌های شعر معاصر ایران صحبت کنم و بدانم "
        "کدام شاعران را پیشنهاد می‌کنید تا بیشتر مطالعه کنم."
    ),
}


def _select_onnx_model_path(onnx_dir: Path) -> Path:
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    preferred = onnx_dir / "model.onnx"
    if preferred.exists():
        return preferred

    candidates = sorted(onnx_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx model files found in {onnx_dir}")
    return candidates[0]



def _build_empty_past_inputs(
    session: ort.InferenceSession, batch_size: int, num_heads: int, head_dim: int
) -> dict[str, np.ndarray]:
    past_inputs: dict[str, np.ndarray] = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        if "past" not in name:
            continue

        shape_len = len(input_meta.shape)
        if shape_len == 5:
            array_shape = (2, batch_size, num_heads, 0, head_dim)
        elif shape_len == 4:
            array_shape = (batch_size, num_heads, 0, head_dim)
        else:
            raise ValueError(
                f"Unsupported past input shape for '{name}': {input_meta.shape}"
            )
        past_inputs[name] = np.zeros(array_shape, dtype=np.float32)
    return past_inputs



def _prepare_onnx_inputs(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    position_ids: np.ndarray | None,
    batch_size: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    onnx_inputs: dict[str, np.ndarray] = {}
    empty_past = _build_empty_past_inputs(session, batch_size, num_heads, head_dim)

    for input_meta in session.get_inputs():
        name = input_meta.name
        if name == "input_ids":
            onnx_inputs[name] = input_ids
        elif name == "attention_mask":
            onnx_inputs[name] = attention_mask
        elif name == "position_ids" and position_ids is not None:
            onnx_inputs[name] = position_ids
        elif name in empty_past:
            onnx_inputs[name] = empty_past[name]
        else:
            raise KeyError(f"Unexpected ONNX input: {name}")
    return onnx_inputs



def validate() -> int:
    np.random.seed(0)
    torch.manual_seed(0)

    tokenizer_pt = AutoTokenizer.from_pretrained(TORCH_MODEL_ID)
    model_pt = AutoModelForCausalLM.from_pretrained(TORCH_MODEL_ID)
    model_pt.eval()

    tokenizer_onnx = AutoTokenizer.from_pretrained(ONNX_DIR)
    onnx_model_path = _select_onnx_model_path(ONNX_DIR)
    session = ort.InferenceSession(onnx_model_path.as_posix(), providers=["CPUExecutionProvider"])

    config = model_pt.config
    num_heads = getattr(config, "n_head", None)
    hidden_size = getattr(config, "hidden_size", None)
    if num_heads is None or hidden_size is None:
        raise ValueError("Model configuration missing required attributes (n_head, hidden_size)")
    head_dim = hidden_size // num_heads

    results = []
    failures = []

    for prompt_id, prompt in PROMPTS.items():
        encoded_pt = tokenizer_pt(prompt, return_tensors="pt")
        encoded_onnx = tokenizer_onnx(prompt, return_tensors="np")

        if not np.array_equal(encoded_pt["input_ids"].cpu().numpy(), encoded_onnx["input_ids"]):
            raise ValueError(f"Tokenizer mismatch for prompt '{prompt_id}'")

        input_ids_pt = encoded_pt["input_ids"]
        attention_mask_pt = encoded_pt.get("attention_mask")
        position_ids_pt = encoded_pt.get("position_ids")

        with torch.no_grad():
            outputs = model_pt(**encoded_pt)
        logits_pt = outputs.logits[:, -1, :]
        argmax_pt = torch.argmax(logits_pt, dim=-1).item()

        input_ids_np = input_ids_pt.cpu().numpy().astype(np.int64)
        attention_mask_np = (
            attention_mask_pt.cpu().numpy().astype(np.int64)
            if attention_mask_pt is not None
            else np.ones_like(input_ids_np, dtype=np.int64)
        )
        position_ids_np = (
            position_ids_pt.cpu().numpy().astype(np.int64)
            if position_ids_pt is not None
            else None
        )

        onnx_inputs = _prepare_onnx_inputs(
            session,
            input_ids=input_ids_np,
            attention_mask=attention_mask_np,
            position_ids=position_ids_np,
            batch_size=input_ids_np.shape[0],
            num_heads=num_heads,
            head_dim=head_dim,
        )

        outputs_onnx = session.run(None, onnx_inputs)
        logits_onnx = outputs_onnx[0]
        logits_onnx_last = logits_onnx[:, -1, :]
        argmax_onnx = int(np.argmax(logits_onnx_last, axis=-1).item())

        logits_pt_np = logits_pt.cpu().numpy()
        max_abs_diff = float(np.max(np.abs(logits_pt_np - logits_onnx_last)))
        argmax_equal = argmax_pt == argmax_onnx

        results.append(
            {
                "prompt_id": prompt_id,
                "argmax_equal": argmax_equal,
                "max_abs_diff": max_abs_diff,
            }
        )

        if not argmax_equal:
            failures.append(
                f"Argmax mismatch for '{prompt_id}': torch={argmax_pt}, onnx={argmax_onnx}"
            )
        if max_abs_diff > MAX_DIFF_THRESHOLD:
            failures.append(
                f"Max abs diff {max_abs_diff:.6f} exceeds threshold for '{prompt_id}'"
            )

    print("prompt_id | argmax_equal | max_abs_diff")
    for row in results:
        print(f"{row['prompt_id']:>8} | {str(row['argmax_equal']):>12} | {row['max_abs_diff']:.6f}")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        return 2

    print("All validations passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(validate())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
