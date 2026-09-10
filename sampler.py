import os
import sys
import time
import math
import argparse
import torch
import jinja2
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_normalized_entropy(probs, top_n):
    """
    Computes Shannon entropy over the top-n tokens, normalized to [0, 1] by log(n).
    T_n(w) = top-n(q_w)
    q_tilde(v) = q_w(v) / sum(q_w(u))
    H_hat(w) = -sum(q_tilde(v) * log(q_tilde(v))) / log(top_n)
    """
    k = min(top_n, probs.shape[-1])
    top_n_probs, _ = torch.topk(probs, k, dim=-1)
    top_n_probs_renorm = top_n_probs / (top_n_probs.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -torch.sum(top_n_probs_renorm * torch.log(top_n_probs_renorm + 1e-12), dim=-1)
    norm_entropy = entropy / math.log(k if k > 1 else 2)
    return norm_entropy


def compute_alpha_weights(alpha):
    """
    Crossfader exponents from Count Bayesie:
    a = 1 - max(0, alpha)
    b = 1 - max(0, -alpha)
    where alpha in [-1, 1]:
      alpha = -1.0 -> a=1, b=0 (pure probability / greedy)
      alpha =  0.0 -> a=1, b=1 (balanced future-entropy: p(w|c) * H_hat(w))
      alpha = +1.0 -> a=0, b=1 (pure future-entropy: H_hat(w))
    """
    alpha = max(-1.0, min(1.0, alpha))
    a = 1.0 - max(0.0, alpha)
    b = 1.0 - max(0.0, -alpha)
    return a, b


def format_instruct_prompt(llm, user_prompt, skip_thought=True):
    """
    Formats an instruction prompt using the model's chat template from GGUF metadata.
    For reasoning models (like Gemma 4), optionally closes the thought channel so the
    model proceeds straight to generating the creative story.
    """
    tmpl = llm.metadata.get('tokenizer.chat_template')
    if tmpl:
        try:
            t = jinja2.Template(tmpl)
            formatted = t.render(
                messages=[{'role': 'user', 'content': user_prompt}],
                add_generation_prompt=True,
                bos_token='',
                eos_token=''
            )
            if skip_thought:
                if formatted.endswith('<|turn>model\n'):
                    formatted += '<|channel>thought\n<channel|>'
                elif formatted.endswith('<think>\n'):
                    formatted += '</think>\n'
            return formatted
        except Exception:
            pass

    # Generic ChatML fallback if instruct requested but template missing
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def future_entropy_sampler(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=80, 
    cand_k=8, 
    top_n_future=10, 
    wavelength=12.0, 
    amp=0.65, 
    min_p=0.01,
    alpha_constant=None,
    sample=False,
    verbose_steps=False,
    stream=False,
    stream_callback=None,
    return_metrics=False,
):
    """
    Generates text using the future-entropy sampler with alpha-wave rhythmic decoding
    using PyTorch / HuggingFace Transformers.
    """
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = input_ids.shape[1]
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            w_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 1. Top cand_k candidates for w with min_p filtering
            k = min(cand_k, w_probs.shape[-1])
            top_k_w_probs, top_k_w_indices = torch.topk(w_probs, k)
            
            mask = top_k_w_probs >= min_p
            if not mask.any():
                chosen_w = top_k_w_indices[0]
                if chosen_w.item() == tokenizer.eos_token_id:
                    break
                input_ids = torch.cat([input_ids, chosen_w.unsqueeze(0).unsqueeze(0)], dim=1)
                if stream or stream_callback:
                    piece = tokenizer.decode([chosen_w.item()], skip_special_tokens=True)
                    if callable(stream_callback):
                        stream_callback(piece)
                    elif stream:
                        sys.stdout.write(piece)
                        sys.stdout.flush()
                continue
                
            top_k_w_probs = top_k_w_probs[mask]
            top_k_w_indices = top_k_w_indices[mask]
            
            # 2. Batch forward pass to look ahead into the future q_w
            batched_input_ids = []
            for w in top_k_w_indices:
                batched_input_ids.append(torch.cat([input_ids, w.unsqueeze(0).unsqueeze(0)], dim=1))
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            
            future_outputs = model(batched_input_ids)
            future_logits = future_outputs.logits[:, -1, :]
            future_probs = torch.softmax(future_logits, dim=-1)
            
            # 3. Compute normalized entropy H_hat(w)
            normalized_entropies = compute_normalized_entropy(future_probs, top_n_future)
            
            # 4. Calculate alpha for this step
            if alpha_constant is not None:
                alpha = float(alpha_constant)
            else:
                alpha = amp * math.sin(2.0 * math.pi * step / wavelength)
            alpha = max(-1.0, min(1.0, alpha))
            a, b = compute_alpha_weights(alpha)
            
            # 5. Score candidates: s(w) = p(w|c)^a * H_hat(w)^b
            scores = (top_k_w_probs ** a) * (normalized_entropies ** b)
            
            # 6. Candidate selection: argmax (default) or multinomial
            if sample:
                scores_sum = scores.sum()
                probs = scores / scores_sum if scores_sum > 0 else torch.ones_like(scores) / len(scores)
                chosen_idx = torch.multinomial(probs, 1).item()
            else:
                chosen_idx = torch.argmax(scores).item()
                
            chosen_w = top_k_w_indices[chosen_idx]
            
            if verbose_steps:
                tok_str = tokenizer.decode([chosen_w.item()])
                print(f"[Step {step:02d}] alpha={alpha:+.2f} (a={a:.2f}, b={b:.2f}) -> chosen={repr(tok_str)}")
                
            if chosen_w.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, chosen_w.unsqueeze(0).unsqueeze(0)], dim=1)

            if stream or stream_callback:
                piece = tokenizer.decode([chosen_w.item()], skip_special_tokens=True)
                if callable(stream_callback):
                    stream_callback(piece)
                elif stream:
                    sys.stdout.write(piece)
                    sys.stdout.flush()

    num_generated = input_ids.shape[1] - prompt_len
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if return_metrics:
        return final_text, num_generated
    return final_text


def future_entropy_sampler_llama(
    llm, 
    prompt, 
    max_new_tokens=80, 
    cand_k=8, 
    top_n_future=10, 
    wavelength=12.0,
    amp=0.65,
    min_p=0.01,
    alpha_constant=None,
    sample=False,
    instruct=False,
    skip_thought=True,
    verbose_steps=False,
    stream=False,
    stream_callback=None,
    return_metrics=False,
):
    """
    Generates text using the future-entropy sampler with alpha-wave rhythmic decoding
    using llama-cpp-python.
    """
    if instruct:
        formatted_prompt = format_instruct_prompt(llm, prompt, skip_thought=skip_thought)
    else:
        formatted_prompt = prompt

    # MUST pass special=True so special/chat tokens are correctly tokenized
    input_ids = llm.tokenize(formatted_prompt.encode('utf-8'), special=True)
    llm.reset()
    llm.eval(input_ids)
    
    # Probe whether model supports fast in-VRAM KV cache rollback (e.g. RoPE vs M-RoPE)
    use_vram_rollback = True
    saved_n = llm.n_tokens
    try:
        dummy_tok = input_ids[-1] if len(input_ids) > 0 else 1
        llm.eval([dummy_tok])
        llm.n_tokens = saved_n
        llm.eval([dummy_tok])
        llm.n_tokens = saved_n
    except Exception:
        use_vram_rollback = False
        llm.reset()
        llm.eval(input_ids)

    # Collect stop tokens
    stop_tokens = {llm.token_eos()}
    for s in ['<turn|>', '<|im_end|>', '<|eot_id|>', '<end_of_turn>', '</s>', '<eos>']:
        try:
            toks = llm.tokenize(s.encode('utf-8'), special=True, add_bos=False)
            if len(toks) == 1:
                stop_tokens.add(toks[0])
        except Exception:
            pass

    generated_tokens = []
    
    for step in range(max_new_tokens):
        logits = torch.tensor(llm._scores[-1, :])
        w_probs = torch.softmax(logits, dim=-1)
        
        # Candidate selection: top cand_k candidates with min_p filtering
        k = min(cand_k, len(w_probs))
        topk = torch.topk(w_probs, k)
        mask = topk.values >= min_p
        if not mask.any():
            chosen_w = topk.indices[0].item()
            if chosen_w in stop_tokens:
                break
            generated_tokens.append(chosen_w)
            llm.eval([chosen_w])
            if stream or stream_callback:
                piece = llm.detokenize([chosen_w]).decode('utf-8', errors='replace')
                if callable(stream_callback):
                    stream_callback(piece)
                elif stream:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
            continue

        top_k_w_probs = topk.values[mask]
        top_k_w_indices = topk.indices[mask]
        
        # Speculative lookahead: in-VRAM rollback if supported, else save_state/load_state
        normalized_entropies = []
        if use_vram_rollback:
            saved_n_tokens = llm.n_tokens
            for w in top_k_w_indices:
                w_idx = w.item()
                llm.eval([w_idx])
                future_logits = torch.tensor(llm._scores[-1, :])
                future_probs = torch.softmax(future_logits, dim=-1)
                norm_entropy = compute_normalized_entropy(future_probs, top_n_future)
                normalized_entropies.append(norm_entropy)
                llm.n_tokens = saved_n_tokens
        else:
            state = llm.save_state()
            for w in top_k_w_indices:
                w_idx = w.item()
                llm.eval([w_idx])
                future_logits = torch.tensor(llm._scores[-1, :])
                future_probs = torch.softmax(future_logits, dim=-1)
                norm_entropy = compute_normalized_entropy(future_probs, top_n_future)
                normalized_entropies.append(norm_entropy)
                llm.load_state(state)
            
        normalized_entropies = torch.stack(normalized_entropies)
        
        # Calculate alpha for this step
        if alpha_constant is not None:
            alpha = float(alpha_constant)
        else:
            alpha = amp * math.sin(2.0 * math.pi * step / wavelength)
        alpha = max(-1.0, min(1.0, alpha))
        a, b = compute_alpha_weights(alpha)
        
        # Score candidates: s(w) = p(w|c)^a * H_hat(w)^b
        scores = (top_k_w_probs ** a) * (normalized_entropies ** b)
        
        # Candidate selection: argmax (default) or multinomial
        if sample:
            scores_sum = scores.sum()
            probs = scores / scores_sum if scores_sum > 0 else torch.ones_like(scores) / len(scores)
            chosen_idx = torch.multinomial(probs, 1).item()
        else:
            chosen_idx = torch.argmax(scores).item()
            
        chosen_w = top_k_w_indices[chosen_idx].item()
        
        if verbose_steps:
            tok_str = llm.detokenize([chosen_w]).decode('utf-8', errors='ignore')
            print(f"[Step {step:02d}] alpha={alpha:+.2f} (a={a:.2f}, b={b:.2f}) -> chosen={repr(tok_str)}")
            
        if chosen_w in stop_tokens:
            break

        generated_tokens.append(chosen_w)
        llm.eval([chosen_w])

        if stream or stream_callback:
            piece = llm.detokenize([chosen_w]).decode('utf-8', errors='replace')
            if callable(stream_callback):
                stream_callback(piece)
            elif stream:
                sys.stdout.write(piece)
                sys.stdout.flush()
            
    output_text = llm.detokenize(generated_tokens).decode('utf-8', errors='ignore')
    
    # Strip any trailing stop marker
    for s in ['<turn|>', '<|im_end|>', '<|eot_id|>', '<end_of_turn>', '</s>', '<eos>']:
        if output_text.endswith(s):
            output_text = output_text[:-len(s)]
            
    final_text = output_text.strip() if instruct else (prompt + output_text)
    if return_metrics:
        return final_text, len(generated_tokens)
    return final_text


def get_model_size_bytes(model_path):
    """
    Returns total size in bytes of the model, summing parts if multi-part GGUF.
    """
    if os.path.isdir(model_path):
        total = 0
        for root, _, files in os.walk(model_path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    elif os.path.isfile(model_path):
        import re
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model_path)
        if match:
            prefix = model_path[: match.start()]
            dirname = os.path.dirname(model_path)
            total = 0
            if os.path.exists(dirname):
                for f in os.listdir(dirname):
                    full = os.path.join(dirname, f)
                    if full.startswith(prefix) and full.endswith(".gguf") and os.path.isfile(full):
                        total += os.path.getsize(full)
            return total if total > 0 else os.path.getsize(model_path)
        return os.path.getsize(model_path)
    return 0


def format_size(num_bytes):
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.1f} GB"
    elif num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    return f"{num_bytes} B"


def calculate_auto_gpu_layers(model_path, requested_layers=-1, n_ctx=2048):
    """
    If requested_layers is -1, inspects model size against available GPU VRAM.
    If the model exceeds VRAM, calculates the maximum safe number of layers to offload.
    """
    if requested_layers != -1:
        return requested_layers

    if not torch.cuda.is_available():
        return 0

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        return -1

    model_bytes = get_model_size_bytes(model_path)
    model_gb = model_bytes / (1024 ** 3)

    # Reserve 3.5 GB for context / KV cache / compute buffers
    usable_vram_gb = max(0.0, free_gb - 3.5)

    if model_gb <= usable_vram_gb:
        return -1  # Fits completely in GPU VRAM

    # Model exceeds free VRAM. Inspect block_count from GGUF metadata
    meta = get_gguf_metadata(model_path)
    arch = meta.get("general.architecture")
    block_count = 32
    if arch and f"{arch}.block_count" in meta:
        block_count = int(meta[f"{arch}.block_count"])

    ratio = max(0.0, usable_vram_gb / max(1.0, model_gb))
    safe_layers = max(1, int(block_count * ratio))
    print(f"\n[VRAM Guard] Model ({model_gb:.1f} GB) exceeds free VRAM ({free_gb:.1f} GB).")
    print(f"[VRAM Guard] Auto-tuning offload to {safe_layers}/{block_count} layers (hybrid GPU/CPU execution).\n")
    return safe_layers


def get_gguf_metadata(model_path, max_kv=120):
    """
    Fast binary reader to extract key-value pairs from GGUF metadata
    without parsing the tensor dictionary or memory-mapping multi-gigabyte files.
    """
    import struct
    meta = {}
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return meta
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(kv_count, max_kv)):
                kl = struct.unpack("<Q", f.read(8))[0]
                k = f.read(kl).decode("utf-8", errors="ignore")
                vt = struct.unpack("<I", f.read(4))[0]
                if vt == 8:  # string
                    sl = struct.unpack("<Q", f.read(8))[0]
                    meta[k] = f.read(sl).decode("utf-8", errors="ignore")
                elif vt == 4:  # uint32
                    meta[k] = struct.unpack("<I", f.read(4))[0]
                elif vt == 5:  # int32
                    meta[k] = struct.unpack("<i", f.read(4))[0]
                elif vt == 10:  # uint64
                    meta[k] = struct.unpack("<Q", f.read(8))[0]
                elif vt == 11:  # int64
                    meta[k] = struct.unpack("<q", f.read(8))[0]
                elif vt == 6:  # float32
                    meta[k] = struct.unpack("<f", f.read(4))[0]
                elif vt == 7:  # bool
                    meta[k] = struct.unpack("<?", f.read(1))[0]
                else:
                    break
    except Exception:
        pass
    return meta


def get_gguf_architecture(model_path):
    """
    Returns the `general.architecture` string from GGUF metadata.
    """
    meta = get_gguf_metadata(model_path, max_kv=40)
    return meta.get("general.architecture")


def load_llama_model(model_path, n_ctx=2048, n_gpu_layers=-1, seed=None):
    """
    Loads a GGUF model with llama.cpp, applying automatic safe layer offloading
    if the model exceeds physical GPU memory.
    """
    from llama_cpp import Llama
    arch = get_gguf_architecture(model_path)

    layers = calculate_auto_gpu_layers(model_path, requested_layers=n_gpu_layers, n_ctx=n_ctx)
    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=layers,
            seed=seed if seed is not None else 42,
            logits_all=True,
            verbose=False,
        )
    except ValueError as e:
        if layers > 0:
            print(f"[VRAM Guard] Offload with {layers} layers failed; falling back to CPU (n_gpu_layers=0)...")
            try:
                return Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=0,
                    seed=seed if seed is not None else 42,
                    logits_all=True,
                    verbose=False,
                )
            except ValueError:
                pass

        if arch:
            if arch == "muse-glimmer":
                raise RuntimeError(
                    f"Failed to load GGUF model '{model_path}' (architecture: 'muse-glimmer'). "
                    f"The 'muse-glimmer' architecture was added in llama.cpp build b10353. "
                    f"Your llama-cpp-python package must be version 0.3.35+ compiled against llama.cpp b10353+ to load this model."
                ) from e
            raise RuntimeError(
                f"Failed to load GGUF model '{model_path}' (architecture: '{arch}'). "
                f"Ensure the model file is complete and supported by llama.cpp."
            ) from e
        raise e


def list_local_models():
    """
    Scans standard local caches across platforms (Linux, macOS, Windows)
    and optional user directories (MODELS_DIR) for usable GGUF or Transformers models,
    excluding multimodal projectors (mmproj), non-language architectures, and non-primary split GGUF shards.
    """
    import re
    models = []
    seen_paths = set()

    def add_model(name, path, mtype):
        real = os.path.realpath(path)
        if real not in seen_paths and os.path.exists(path):
            seen_paths.add(real)
            models.append({"name": name, "path": path, "type": mtype})

    def process_gguf_file(f, full_path, display_prefix):
        # Skip multimodal projectors
        if f.startswith("mmproj") or "mmproj" in f:
            return

        # Skip known non-LLM architectures (e.g. image diffusion models)
        arch = get_gguf_architecture(full_path)
        if arch in ["lumina2", "qwen_image", "flux", "diffusion", "wan"]:
            return  # Skip image/diffusion models

        # If multi-part split GGUF (e.g. -00002-of-00005.gguf), only keep the first shard (-00001-of-)
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f)
        if match:
            part_num = int(match.group(1))
            total_parts = int(match.group(2))
            if part_num != 1:
                return  # Skip secondary shards
            total_size = get_model_size_bytes(full_path)
            clean_name = f[: match.start()]
            add_model(f"{display_prefix} ({clean_name}, {total_parts} parts, {format_size(total_size)})", full_path, "gguf")
        else:
            size_str = format_size(os.path.getsize(full_path))
            add_model(f"{display_prefix} ({size_str})", full_path, "gguf")

    # 1. Custom / local directories via environment variable or current ./models dir
    custom_dirs = []
    if os.environ.get("MODELS_DIR"):
        custom_dirs.append(os.environ["MODELS_DIR"])
    if os.environ.get("MODEL_DIRS"):
        custom_dirs.extend(os.environ["MODEL_DIRS"].split(os.pathsep))
    if os.path.isdir("models"):
        custom_dirs.append(os.path.abspath("models"))

    for cdir in custom_dirs:
        cdir = os.path.expanduser(cdir)
        if os.path.isdir(cdir):
            for root, _, files in os.walk(cdir):
                for f in files:
                    if f.endswith(".gguf"):
                        rel = os.path.relpath(os.path.join(root, f), cdir)
                        process_gguf_file(f, os.path.join(root, f), f"[Local Dir] {rel}")
                if os.path.exists(os.path.join(root, "config.json")):
                    rel = os.path.relpath(root, cdir)
                    dir_size = get_model_size_bytes(root)
                    add_model(f"[Local Transformers] {rel} ({format_size(dir_size)})", root, "transformers")

    # 2. HuggingFace Cache (respects HF_HOME if set)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_cache_dir = os.path.join(os.path.expanduser(hf_home), "hub")
    else:
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    if os.path.exists(hf_cache_dir):
        for item in os.listdir(hf_cache_dir):
            if item.startswith("models--"):
                model_name = item.replace("models--", "").replace("--", "/")
                snapshots_dir = os.path.join(hf_cache_dir, item, "snapshots")
                if os.path.exists(snapshots_dir):
                    for commit_hash in os.listdir(snapshots_dir):
                        snap_path = os.path.join(snapshots_dir, commit_hash)
                        found_gguf = False
                        for root, _, files in os.walk(snap_path):
                            for f in files:
                                if f.endswith(".gguf"):
                                    rel = os.path.relpath(os.path.join(root, f), snap_path)
                                    process_gguf_file(f, os.path.join(root, f), f"[HF GGUF] {model_name} ({rel})")
                                    found_gguf = True
                        if not found_gguf and os.path.exists(os.path.join(snap_path, "config.json")):
                            dir_size = get_model_size_bytes(snap_path)
                            add_model(f"[HF Transformers] {model_name} ({format_size(dir_size)})", snap_path, "transformers")

    # 3. LM Studio Caches (Linux, macOS, Windows)
    lm_candidates = [
        os.path.expanduser("~/.cache/lm-studio/models"),
        os.path.expanduser("~/Library/Application Support/LM-Studio/models"),
        os.path.expanduser("~/.lmstudio/models"),
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        lm_candidates.append(os.path.join(appdata, "LM-Studio", "models"))

    for lm_studio_dir in lm_candidates:
        if os.path.exists(lm_studio_dir):
            for root, _, files in os.walk(lm_studio_dir):
                for file in files:
                    if file.endswith(".gguf"):
                        rel_path = os.path.relpath(os.path.join(root, file), lm_studio_dir)
                        process_gguf_file(file, os.path.join(root, file), f"[LM Studio] {rel_path}")

    # 4. GPT4All Caches (Linux, macOS, Windows)
    gpt4all_candidates = [
        os.path.expanduser("~/.local/share/nomic.ai/GPT4All/"),
        os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All/"),
    ]
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        gpt4all_candidates.append(os.path.join(localappdata, "nomic.ai", "GPT4All"))

    for gpt4all_dir in gpt4all_candidates:
        if os.path.exists(gpt4all_dir):
            for file in os.listdir(gpt4all_dir):
                if file.endswith(".gguf"):
                    process_gguf_file(file, os.path.join(gpt4all_dir, file), f"[GPT4All] {file}")

    # 5. Ollama models directory (OLLAMA_MODELS or default)
    ollama_dir = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
    if os.path.exists(ollama_dir):
        for root, _, files in os.walk(ollama_dir):
            for file in files:
                if file.endswith(".gguf"):
                    rel_path = os.path.relpath(os.path.join(root, file), ollama_dir)
                    process_gguf_file(file, os.path.join(root, file), f"[Ollama] {rel_path}")

    return sorted(models, key=lambda x: x["name"])


def main():
    parser = argparse.ArgumentParser(description="Future-Entropy Sampler (Count Bayesie)")
    parser.add_argument("--model_path", "--model", dest="model_path", type=str, help="Path to local safetensors directory or GGUF file")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a futuristic city,", help="Input prompt")
    parser.add_argument("--instruct", action="store_true", help="Format prompt with model chat template")
    parser.add_argument("--raw", "--no-instruct", dest="raw", action="store_true", help="Force raw prompt completion without chat template")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of tokens to generate")
    parser.add_argument("--cand_k", type=int, default=8, help="Candidates k to evaluate at each step (default: 8)")
    parser.add_argument("--top_n", type=int, default=10, help="Top n future tokens for entropy (default: 10)")
    parser.add_argument("--wavelength", type=float, default=12.0, help="Wavelength of alpha sine wave in tokens (default: 12.0)")
    parser.add_argument("--amp", type=float, default=0.65, help="Amplitude for alpha wave (default: 0.65)")
    parser.add_argument("--min_p", type=float, default=0.01, help="Minimum candidate probability to evaluate (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha in [-1.0, 1.0] to test static crossfader without wave")
    parser.add_argument("--sample", action="store_true", help="Use stochastic multinomial sampling instead of argmax")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--verbose_steps", action="store_true", help="Print alpha and selected tokens at each step")
    parser.add_argument("--no_stream", action="store_true", help="Disable real-time token streaming")
    parser.add_argument("--compare", action="store_true", help="Run comparative benchmark (Greedy vs Temperature vs Alpha-Wave)")
    args, unknown = parser.parse_known_args()
    
    if args.compare:
        import compare
        # Remove --compare from sys.argv and invoke compare.main()
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--compare"]
        compare.main()
        return

    model_path = args.model_path
    
    if not model_path:
        print("No model path provided. Scanning local caches...")
        models = list_local_models()
        if not models:
            print("No models found in HuggingFace, LM Studio, or GPT4All caches. Please provide a path using --model_path.")
            exit(1)
            
        print("\nAvailable models:")
        for i, m in enumerate(models):
            print(f"[{i + 1}] {m['name']}")
            
        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    model_path = models[choice_idx]['path']
                    print(f"Selected: {models[choice_idx]['name']}")
                    break
                print("Invalid selection.")
            except (ValueError, KeyboardInterrupt, EOFError):
                if choice.lower() in ['q', 'quit', 'exit']:
                    exit(0)
                print("Please enter a valid number or 'q' to quit.")

    print(f"Loading model from {model_path}...")
    prompt = args.prompt
    print(f"Prompt: {prompt}")
    
    # Auto-detect if instruct mode is recommended
    instruct = args.instruct
    if not args.raw and not instruct:
        is_it_model = any(sig in model_path.lower() for sig in ['-it', 'instruct', 'chat'])
        is_gemma = 'gemma' in model_path.lower()
        if is_gemma or is_it_model:
            instruct = True
            print("Auto-detected instruct-tuned model. Enabling chat template formatting (pass --raw to disable).")

    stream = not args.no_stream

    if os.path.isfile(model_path) and model_path.endswith('.gguf'):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("Please install llama-cpp-python to run GGUF files.")
            exit(1)
            
        print(f"Loading GGUF model into llama.cpp (requested n_gpu_layers={args.n_gpu_layers})...")
        llm = load_llama_model(
            model_path=model_path, 
            n_ctx=2048, 
            n_gpu_layers=args.n_gpu_layers, 
        )
        if not stream:
            print("Generating text with future-entropy sampler...")
        else:
            if not instruct:
                if sys.stdout.isatty():
                    sys.stdout.write(f"\033[90m{prompt}\033[0m")
                else:
                    sys.stdout.write(prompt)
                sys.stdout.flush()

        t0 = time.time()
        result, num_tokens = future_entropy_sampler_llama(
            llm, 
            prompt, 
            max_new_tokens=args.max_new_tokens,
            cand_k=args.cand_k,
            top_n_future=args.top_n,
            wavelength=args.wavelength,
            amp=args.amp,
            min_p=args.min_p,
            alpha_constant=args.alpha,
            sample=args.sample,
            instruct=instruct,
            verbose_steps=args.verbose_steps,
            stream=stream,
            return_metrics=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float16
        else:
            device_map = None
            dtype = torch.float32

        print(f"Using device_map={device_map}, dtype={dtype} for transformers...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,
            device_map=device_map,
            torch_dtype=dtype
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not stream:
            print("Generating text with future-entropy sampler using transformers...")
        else:
            if not instruct:
                if sys.stdout.isatty():
                    sys.stdout.write(f"\033[90m{prompt}\033[0m")
                else:
                    sys.stdout.write(prompt)
                sys.stdout.flush()

        t0 = time.time()
        result, num_tokens = future_entropy_sampler(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=args.max_new_tokens,
            cand_k=args.cand_k,
            top_n_future=args.top_n,
            wavelength=args.wavelength,
            amp=args.amp,
            min_p=args.min_p,
            alpha_constant=args.alpha,
            sample=args.sample,
            verbose_steps=args.verbose_steps,
            stream=stream,
            return_metrics=True,
        )
        
    elapsed = time.time() - t0
    tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0

    if stream:
        print(f"\n\n[Finished: {num_tokens} tokens generated in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]\n")
    else:
        print("\nResult:\n")
        print(result)
        print(f"\n[Finished: {num_tokens} tokens generated in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]\n")


if __name__ == "__main__":
    main()
