import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from datasets import load_dataset
from typing import Dict, Any

from src.utils.data_utils import HHDatasetProcessor
from src.utils.safety_integration import SafetyFilter


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_overrides(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge for simple overrides (e.g., colab_config)."""
    out = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and key in out and isinstance(out[key], dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


class GemmaQLoRATrainer:
    def __init__(self, config_path: str, override_path: str | None = None):
        base = load_config(config_path)
        if override_path and os.path.exists(override_path):
            override = load_config(override_path)
            self.cfg = merge_overrides(base, override)
        else:
            self.cfg = base

    def setup_model_and_tokenizer(self):
        mcfg = self.cfg['model']

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=mcfg['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, mcfg['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=mcfg['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=mcfg['bnb_4bit_use_double_quant'],
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            mcfg['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(mcfg['name'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # PEFT - LoRA
        lcfg = self.cfg['lora']
        lora = LoraConfig(
            r=lcfg['r'],
            lora_alpha=lcfg['lora_alpha'],
            lora_dropout=lcfg['lora_dropout'],
            target_modules=lcfg['target_modules'],
            bias=lcfg['bias'],
            task_type=lcfg['task_type'],
        )
        model = get_peft_model(model, lora)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        return model, tokenizer

    def prepare_datasets(self, tokenizer, safety_filter: SafetyFilter | None):
        proc = HHDatasetProcessor(self.cfg, tokenizer, safety_filter)
        train_ds = proc.load_split(self.cfg['dataset']['train_split'])
        eval_ds = proc.load_split(self.cfg['dataset']['eval_split'])

        # Optional safety filter on train
        if self.cfg.get('safety', {}).get('enabled', False) and safety_filter is not None:
            train_ds = proc.apply_safety_filter(train_ds)

        train_tok = train_ds.map(
            proc.preprocess_batch, batched=True, remove_columns=train_ds.column_names
        )
        eval_tok = eval_ds.map(
            proc.preprocess_batch, batched=True, remove_columns=eval_ds.column_names
        )
        return train_tok, eval_tok

    def train(self):
        # W&B (delay import to allow offline use if desired)
        import wandb
        wcfg = self.cfg['wandb']
        wandb.login()
        wandb.init(project=wcfg['project'], entity=wcfg['entity'], tags=wcfg.get('tags', []), config=self.cfg)

        # Setup model
        model, tokenizer = self.setup_model_and_tokenizer()

        # Safety classifier (optional)
        scfg = self.cfg.get('safety', {})
        safety_filter = None
        if scfg.get('enabled', False):
            safety_filter = SafetyFilter(
                classifier_config_path=scfg['classifier_config_path'],
                checkpoint_dir=scfg['checkpoint_dir'],
            )

        # Data
        train_ds, eval_ds = self.prepare_datasets(tokenizer, safety_filter)

        # Training args
        tcfg = self.cfg['training']
        args = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=tcfg['batch_size'],
            per_device_eval_batch_size=tcfg['batch_size'],
            gradient_accumulation_steps=tcfg['gradient_accumulation_steps'],
            num_train_epochs=tcfg['num_epochs'],
            learning_rate=tcfg['learning_rate'],
            warmup_steps=tcfg['warmup_steps'],
            logging_steps=tcfg['logging_steps'],
            save_steps=tcfg['save_steps'],
            eval_steps=tcfg['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=tcfg['fp16'],
            gradient_checkpointing=tcfg['gradient_checkpointing'],
            optim=tcfg['optim'],
            report_to=["wandb"],
            run_name="gemma-7b-helpful-qlora",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )

        trainer.train()
        trainer.save_model("./final_model")
        try:
            model.save_pretrained("./lora_adapters")
        except Exception:
            pass

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    trainer = GemmaQLoRATrainer(args.config, args.override)
    trainer.train()
