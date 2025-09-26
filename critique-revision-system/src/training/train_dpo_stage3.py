#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer
from peft import PeftModel

# Disable W&B and reduce noisy logs by default
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 DPO training on critique/revision pairs.")
    p.add_argument("--repo-root", type=str, default=".", help="Repository root (for relative artifacts).")
    p.add_argument("--pairs-path", type=str, required=True, help="Path to JSONL pairs file from critique-revision generation.")
    p.add_argument("--base-model-id", type=str, default="google/gemma-2b-it", help="Base HF model id.")
    p.add_argument("--stage2-adapter-path", type=str, required=True, help="Path to Stage 2 LoRA adapters.")
    p.add_argument("--output-dir", type=str, required=True, help="Output dir for Stage 3 artifacts.")

    # Training hyperparameters
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epoch-based training if > 0.")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (preference strength).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)

    # Memory controls
    p.add_argument("--cpu-ref-model", action="store_true", help="Load reference model on CPU to save VRAM.")
    return p.parse_args()


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_pairs(pairs_path: Path) -> List[Dict]:
    pairs: List[Dict] = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {ln} of {pairs_path}: {e}")
            pairs.append(obj)
    if not pairs:
        raise ValueError(f"No pairs found in {pairs_path}")
    return pairs


def extract_prompt_initial_revised(example: Dict) -> Tuple[str, str, str]:
    # Harmonize keys from critique_revision.py output
    prompt_keys = ["prompt", "instruction", "input", "user_prompt"]
    initial_keys = [
        "base_response",  # critique_revision.py uses this
        "initial_response",
        "initial",
        "draft",
        "raw_response",
    ]
    revised_keys = ["revised_response", "revision", "final_response", "final"]

    prompt = next((example.get(k) for k in prompt_keys if example.get(k)), None)
    initial = next((example.get(k) for k in initial_keys if example.get(k)), None)
    revised = next((example.get(k) for k in revised_keys if example.get(k)), None)

    if prompt is None or initial is None or revised is None:
        raise KeyError(
            f"Missing required fields. Found prompt={prompt is not None}, initial={initial is not None}, revised={revised is not None}. "
            f"Available keys: {list(example.keys())}"
        )
    return prompt, initial, revised


def to_dpo_records(pairs: List[Dict]) -> List[Dict]:
    recs: List[Dict] = []
    for ex in pairs:
        prompt, initial, revised = extract_prompt_initial_revised(ex)
        # If the generator already decided the better one, honor it
        chosen_field = ex.get("chosen")
        if chosen_field == "revised":
            chosen = revised
            rejected = initial
        elif chosen_field == "base":
            chosen = initial
            rejected = revised
        else:
            # Fallback: prefer revised by assumption if no explicit choice
            chosen = revised
            rejected = initial

        prompt_str = f"User: {prompt}\nAssistant:"
        recs.append({"prompt": prompt_str, "chosen": chosen, "rejected": rejected})
    return recs


def main() -> None:
    setup_logger()
    args = parse_args()

    # Seeds for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(args.repo_root).resolve()
    pairs_path = Path(args.pairs_path).resolve()
    out_dir = Path(args.output_dir).resolve()
    stage2_adapter_path = Path(args.stage2_adapter_path).resolve()

    logging.info(f"Repo root: {repo_root}")
    logging.info(f"Pairs path: {pairs_path}")
    logging.info(f"Stage 2 adapters: {stage2_adapter_path}")
    logging.info(f"Output dir: {out_dir}")

    ensure_exists(repo_root, "Repository root")
    ensure_exists(pairs_path, "Pairs path")
    ensure_exists(stage2_adapter_path, "Stage 2 adapter path")

    # Prepare output directories
    checkpoints_dir = out_dir / "checkpoints"
    lora_out_dir = out_dir / "models" / "lora_adapters"
    dpo_ds_out = out_dir / "dpo_dataset" / "train.jsonl"
    logs_dir = out_dir / "logs"
    for p in (checkpoints_dir, lora_out_dir, dpo_ds_out.parent, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Load and convert dataset
    pairs = load_pairs(pairs_path)
    dpo_rows = to_dpo_records(pairs)
    with dpo_ds_out.open("w", encoding="utf-8") as f:
        for r in dpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    train_ds = Dataset.from_list(dpo_rows)
    logging.info(f"Prepared DPO dataset with {len(train_ds)} samples -> {dpo_ds_out}")

    # Tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True, trust_remote_code=True)
    ensure_pad_token(tokenizer)

    # Use FP16 on CUDA by default (T4-friendly); otherwise float32
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map: Optional[str] = "auto" if torch.cuda.is_available() else None

    logging.info(f"Loading base model: {args.base_model_id} (dtype={dtype}, device_map={device_map})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load Stage 2 LoRA adapters and make them trainable
    logging.info("Loading Stage 2 LoRA adapters into base model (trainable)...")
    model = PeftModel.from_pretrained(
        base_model,
        str(stage2_adapter_path),
        is_trainable=True,
    )

    # DPO reference model
    ref_model = None
    if args.cpu_ref_model:
        logging.info("Loading reference model on CPU to save VRAM.")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    # Training arguments — W&B disabled
    train_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=(args.max_steps if args.max_steps and args.max_steps > 0 else -1),
        logging_dir=str(logs_dir),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],  # ensure W&B is off
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    logging.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=train_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    logging.info("Starting DPO training...")
    result = trainer.train()
    metrics = result.metrics if result is not None else {}
    logging.info(f"Training complete. Metrics: {metrics}")

    # Save final LoRA adapters
    logging.info(f"Saving LoRA adapters to {lora_out_dir}")
    model.save_pretrained(str(lora_out_dir))
    tokenizer.save_pretrained(str(lora_out_dir))

    # Persist metrics
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Stage 3 DPO training finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error in train_dpo_stage3: {e}")
        sys.exit(1)
