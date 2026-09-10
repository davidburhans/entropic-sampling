import os
import sys
from llama_cpp import Llama
from sampler import list_local_models

def test():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH")
    if not model_path:
        local_ggufs = [m for m in list_local_models() if m["type"] == "gguf"]
        if local_ggufs:
            # Prefer a smaller model if available (e.g., E4B or 12b or any)
            preferred = [m for m in local_ggufs if any(k in m["path"].lower() for k in ["e4b", "12b", "8b", "7b", "3b"])]
            model_path = preferred[0]["path"] if preferred else local_ggufs[0]["path"]
            print(f"No model path specified, using auto-detected model: {model_path}")
        else:
            print("Error: No GGUF model found in local caches.")
            print("Usage: python test_llama.py /path/to/model.gguf  (or set MODEL_PATH=...)")
            sys.exit(1)

    print(f"Loading model {model_path} with GPU acceleration...")
    llm = Llama(model_path=model_path, n_ctx=256, n_gpu_layers=-1, logits_all=True, verbose=False)
    
    prompt = "Once upon a time in a futuristic city,"
    input_ids = llm.tokenize(prompt.encode('utf-8'), special=True)
    
    print("Evaluating prompt...")
    llm.eval(input_ids)
    
    logits = llm._scores[-1, :]
    print("Logits shape:", logits.shape)
    
    original_state = llm.save_state()
    
    # Try one token lookahead
    llm.eval([100])
    future_logits = llm._scores[-1, :]
    print("Future logits shape:", future_logits.shape)
    
    llm.load_state(original_state)
    restored_logits = llm._scores[-1, :]
    print("State loaded successfully. Logits match:", (logits == restored_logits).all())
    
if __name__ == "__main__":
    test()
