import streamlit as st
import os
import sys

# Ensure local src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.generate import GemmaInference
from src.utils.safety_integration import SafetyFilter

st.set_page_config(page_title="Stage 2: Helpful Response Comparison", page_icon="🤖", layout="wide")

@st.cache_resource
def load_resources():
    base = GemmaInference(base_model_name="google/gemma-2b-it", adapter_path=None, load_in_4bit=False)

    # Compute canonical adapters path relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    adapters = os.path.join(repo_root, "artifacts", "stage2_artifacts", "lora_adapters")
    finetuned = GemmaInference(base_model_name="google/gemma-2b-it", adapter_path=adapters, load_in_4bit=False)

    safety = SafetyFilter(
        classifier_config_path="../safety-text-classifier/configs/base_config.yaml",
        checkpoint_dir="../safety-text-classifier/checkpoints/best_model",
    )
    return base, finetuned, safety

def main():
    st.title("🤖 Helpful Response Fine-tuning: Base vs Fine-tuned")
    st.write("Gemma-7B-IT with QLoRA on Anthropic HH. Safety overlay from Stage 1 classifier.")

    base, finetuned, safety = load_resources()

    with st.sidebar:
        st.header("Generation Parameters")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
        max_length = st.slider("Max length", 64, 1024, 512, 32)
        safety_threshold = st.slider("Safety threshold", 0.0, 1.0, 0.8, 0.05)

    st.subheader("Enter a prompt")
    default_examples = [
        "How can I improve my public speaking skills?",
        "Explain transformers to a high school student.",
        "Give me a weekly meal prep plan for a busy grad student.",
        "What steps should I take to get started with machine learning?",
        "What are some techniques to manage stress effectively?",
    ]
    example = st.selectbox("Examples", options=["Custom"] + default_examples)
    prompt = st.text_area("Prompt", value="" if example == "Custom" else example, height=120)

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating..."):
                base_out = base.generate(prompt, max_length=max_length, temperature=temperature, top_p=top_p)
                ft_out = finetuned.generate(prompt, max_length=max_length, temperature=temperature, top_p=top_p)

                base_safety = safety.score_text(base_out)
                ft_safety = safety.score_text(ft_out)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📘 Base (Gemma-7B-IT)")
                st.caption(f"Safety: {'🟢' if base_safety >= safety_threshold else '🔴'} {base_safety:.2f}")
                st.write(base_out)
            with c2:
                st.subheader("📗 Fine-tuned (QLoRA + HH)")
                st.caption(f"Safety: {'🟢' if ft_safety >= safety_threshold else '🔴'} {ft_safety:.2f}")
                st.write(ft_out)

            st.subheader("Quick stats")
            s1, s2, s3 = st.columns(3)
            s1.metric("Base length", len(base_out.split()))
            s2.metric("Fine-tuned length", len(ft_out.split()))
            improvement = (ft_safety - base_safety) if base_safety > 0 else 0.0
            s3.metric("Safety Δ", f"{improvement:.2f}")

if __name__ == "__main__":
    main()
