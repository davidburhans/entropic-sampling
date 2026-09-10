from llama_cpp import Llama
from sampler import future_entropy_sampler_llama

# Use 26B Gemma model on GPU
model_path = "/home/dave/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-qat-q4_0-gguf/snapshots/dfc00409adc70be497fee9c90bfe76b3ee130f2e/gemma-4-26B_q4_0-it.gguf"
print(f"Loading model {model_path} with GPU acceleration...")
llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, logits_all=True, verbose=False)

user_prompt = "Write a creative story starting with: Once upon a time in a futuristic city,"

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
