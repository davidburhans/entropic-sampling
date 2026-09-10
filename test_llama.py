import os
from llama_cpp import Llama

def test():
    model_path = "/home/dave/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/bb3b92e6f031fa438b409f898dd9f14f499a0cb0/gemma-4-E4B_q4_0-it.gguf"
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
