import os
import sys
from llama_cpp import Llama
from sampler import future_entropy_sampler_llama, list_local_models

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH")
    if not model_path:
        local_ggufs = [m for m in list_local_models() if m["type"] == "gguf"]
        if local_ggufs:
            # Prefer Gemma or Qwen instruct models if available
            preferred = [m for m in local_ggufs if any(k in m["name"].lower() for k in ["gemma", "qwen"])]
            model_path = preferred[0]["path"] if preferred else local_ggufs[0]["path"]
            print(f"No model path specified, using auto-detected model: {model_path}")
        else:
            print("Error: No GGUF model found in local caches.")
            print("Usage: python test_generation.py [model_path] [prompt]")
            sys.exit(1)

    print(f"Loading model {model_path} with GPU acceleration...")
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, logits_all=True, verbose=False)

    user_prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a creative story starting with: Once upon a time in a futuristic city,"
    print("\n--- Testing Future Entropy Sampler (Alpha-Wave Rhythmic Decoding) ---")
    result = future_entropy_sampler_llama(
        llm,
        user_prompt,
        max_new_tokens=80,
        cand_k=12,
        top_n_future=10,
        wavelength=12.0,
        amp=1.0,
        instruct=True,
        verbose_steps=True
    )

    print("\n--- Generated Story ---")
    print(result)


if __name__ == "__main__":
    main()
