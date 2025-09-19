import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class GemmaInference:
    def __init__(self, base_model_name: str, adapter_path: str | None = None, load_in_4bit: bool = True):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            if bnb_config is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            # Fallback if bitsandbytes path fails
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        if adapter_path:
            # Prefer loading from a local adapter directory and validate structure
            looks_local = os.path.isabs(adapter_path) or adapter_path.startswith("./") or adapter_path.startswith("../") or adapter_path.startswith("~") or (os.sep in adapter_path)
            if looks_local and not os.path.isdir(adapter_path):
                raise FileNotFoundError(
                    f"Adapter path '{adapter_path}' does not exist or is not a directory."
                )
            if os.path.isdir(adapter_path):
                cfg_path = os.path.join(adapter_path, "adapter_config.json")
                if not os.path.exists(cfg_path):
                    raise FileNotFoundError(
                        f"Expected '{cfg_path}' but it was not found. The LoRA adapters do not appear to be saved.\n"
                        f"- Ensure training completed and saved adapters.\n"
                        f"- After training, you should see: [Stage2] Saved LoRA adapters to ./lora_adapters\n"
                        f"- If not present, re-run training and check for errors during save_pretrained."
                    )
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            else:
                # If not a directory and not clearly local, attempt to treat as a remote model id
                # (No silent fallback: invalid ids will raise clearly.)
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "<start_of_turn>model" in text:
            text = text.split("<start_of_turn>model")[-1].strip()
        return text

    # Expose underlying model/tokenizer if needed for evaluation
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
