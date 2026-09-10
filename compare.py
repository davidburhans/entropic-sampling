import os
import sys
import time
import math
import random
import argparse
import torch

from sampler import (
    compute_normalized_entropy,
    compute_alpha_weights,
    format_instruct_prompt,
    future_entropy_sampler_llama,
    future_entropy_sampler,
    list_local_models,
)


def standard_temperature_sampling_llama(
    llm,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.95,
    seed=42,
    instruct=False,
    skip_thought=True,
):
    """
    Standard next-token temperature + top-p sampling with llama-cpp-python.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    if instruct:
        formatted_prompt = format_instruct_prompt(llm, prompt, skip_thought=skip_thought)
    else:
        formatted_prompt = prompt

    input_ids = llm.tokenize(formatted_prompt.encode("utf-8"), special=True)
    llm.reset()
    llm.eval(input_ids)

    stop_tokens = {llm.token_eos()}
    for s in ["<turn|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>"]:
        try:
            toks = llm.tokenize(s.encode("utf-8"), special=True, add_bos=False)
            if len(toks) == 1:
                stop_tokens.add(toks[0])
        except Exception:
            pass

    generated_tokens = []
    for _ in range(max_new_tokens):
        logits = torch.tensor(llm._scores[-1, :], dtype=torch.float32)
        if temperature <= 1e-4:
            chosen_w = torch.argmax(logits).item()
        else:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else torch.ones_like(probs) / len(probs)
            chosen_w = torch.multinomial(probs, 1).item()

        generated_tokens.append(chosen_w)
        llm.eval([chosen_w])
        if chosen_w in stop_tokens:
            break

    output_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")
    for s in ["<turn|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>"]:
        if output_text.endswith(s):
            output_text = output_text[:-len(s)]

    if instruct:
        return output_text.strip()
    return prompt + output_text


def standard_temperature_sampling_transformers(
    model,
    tokenizer,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.95,
    seed=42,
):
    """
    Standard next-token temperature + top-p sampling with HuggingFace Transformers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]

            if temperature <= 1e-4:
                chosen_w = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                probs_sum = probs.sum()
                probs = probs / probs_sum if probs_sum > 0 else torch.ones_like(probs) / len(probs)
                chosen_w = torch.multinomial(probs, 1).unsqueeze(0)

            input_ids = torch.cat([input_ids, chosen_w], dim=1)
            if chosen_w.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def run_comparison_llama(
    llm,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.95,
    cand_k=12,
    top_n_future=10,
    wavelength=12.0,
    amp=1.0,
    seed=42,
    instruct=False,
):
    """
    Runs all 4 sampling methods under the exact same seed and prompt.
    """
    results = {}

    # 1. Greedy Decoding (temp = 0.0)
    print("  [1/4] Generating with Greedy Decoding (p(w|c) argmax)...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_greedy = standard_temperature_sampling_llama(
        llm, prompt, max_new_tokens=max_new_tokens, temperature=0.0, seed=seed, instruct=instruct
    )
    t_greedy = time.time() - t0
    results["Greedy (Argmax / Temp 0.0)"] = {
        "text": out_greedy,
        "time": t_greedy,
        "description": "Standard greedy decoding (argmax next-token probability). Frequently funnels into clichés and predictable prose.",
    }

    # 2. Standard Temperature Sampling (temp = 0.8)
    print(f"  [2/4] Generating with Standard Temperature Sampling (T={temperature}, top-p={top_p})...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_temp = standard_temperature_sampling_llama(
        llm, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed, instruct=instruct
    )
    t_temp = time.time() - t0
    results[f"Standard Sampling (Temp {temperature}, Top-p {top_p})"] = {
        "text": out_temp,
        "time": t_temp,
        "description": f"Traditional stochastic sampling from the scaled next-token distribution with top-p={top_p}.",
    }

    # 3. Static Future-Entropy (alpha = 0.0)
    print("  [3/4] Generating with Static Future-Entropy (alpha = 0.0 balanced)...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_static = future_entropy_sampler_llama(
        llm,
        prompt,
        max_new_tokens=max_new_tokens,
        cand_k=cand_k,
        top_n_future=top_n_future,
        alpha_constant=0.0,
        sample=False,
        instruct=instruct,
    )
    t_static = time.time() - t0
    results["Static Future-Entropy (alpha = 0.0)"] = {
        "text": out_static,
        "time": t_static,
        "description": "Crossfader balance s(w) = p(w|c) * H_hat(w). Weights high future optionality at every token.",
    }

    # 4. Alpha-Wave Rhythmic Decoding
    print(f"  [4/4] Generating with Alpha-Wave Rhythmic Decoding (wavelength={wavelength}, amp={amp})...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_wave = future_entropy_sampler_llama(
        llm,
        prompt,
        max_new_tokens=max_new_tokens,
        cand_k=cand_k,
        top_n_future=top_n_future,
        wavelength=wavelength,
        amp=amp,
        sample=False,
        instruct=instruct,
    )
    t_wave = time.time() - t0
    results["Alpha-Wave Rhythmic Decoding (Sine Wave)"] = {
        "text": out_wave,
        "time": t_wave,
        "description": f"Sinusoidal oscillation of alpha in [-{amp}, +{amp}] with wavelength={wavelength} tokens. Natural breathing cadence between anchors and creative leaps.",
    }

    return results


def run_comparison_transformers(
    model,
    tokenizer,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.95,
    cand_k=12,
    top_n_future=10,
    wavelength=12.0,
    amp=1.0,
    seed=42,
):
    """
    Runs all 4 sampling methods with HuggingFace Transformers.
    """
    results = {}

    print("  [1/4] Generating with Greedy Decoding (Temp 0.0)...", flush=True)
    t0 = time.time()
    out_greedy = standard_temperature_sampling_transformers(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0, seed=seed
    )
    results["Greedy (Argmax / Temp 0.0)"] = {
        "text": out_greedy,
        "time": time.time() - t0,
        "description": "Standard greedy decoding (argmax next-token probability).",
    }

    print(f"  [2/4] Generating with Standard Temperature Sampling (T={temperature})...", flush=True)
    t0 = time.time()
    out_temp = standard_temperature_sampling_transformers(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed
    )
    results[f"Standard Sampling (Temp {temperature}, Top-p {top_p})"] = {
        "text": out_temp,
        "time": time.time() - t0,
        "description": f"Traditional stochastic sampling with temperature={temperature}.",
    }

    print("  [3/4] Generating with Static Future-Entropy (alpha = 0.0)...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_static = future_entropy_sampler(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        cand_k=cand_k,
        top_n_future=top_n_future,
        alpha_constant=0.0,
        sample=False,
    )
    results["Static Future-Entropy (alpha = 0.0)"] = {
        "text": out_static,
        "time": time.time() - t0,
        "description": "Balanced crossfader s(w) = p(w|c) * H_hat(w).",
    }

    print("  [4/4] Generating with Alpha-Wave Rhythmic Decoding...", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    out_wave = future_entropy_sampler(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        cand_k=cand_k,
        top_n_future=top_n_future,
        wavelength=wavelength,
        amp=amp,
        sample=False,
    )
    results["Alpha-Wave Rhythmic Decoding (Sine Wave)"] = {
        "text": out_wave,
        "time": time.time() - t0,
        "description": f"Sinusoidal alpha wave (wavelength={wavelength}, amp={amp}).",
    }

    return results


def print_comparison(results, prompt, seed):
    """
    Renders a formatted comparison in the terminal.
    """
    width = 80
    sep = "=" * width
    subsep = "-" * width

    print(f"\n{sep}")
    print(" SAMPLING COMPARISON BENCHMARK ".center(width, "#"))
    print(sep)
    print(f"Prompt: {repr(prompt)}")
    print(f"Random Seed: {seed}")
    print(f"{sep}\n")

    for idx, (title, data) in enumerate(results.items(), start=1):
        print(f"[{idx}] {title.upper()}")
        print(f"    Rationale: {data['description']}")
        print(f"    Generation time: {data['time']:.2f}s")
        print(subsep)
        print(data["text"])
        print(f"{subsep}\n")


def save_markdown_report(filepath, results, prompt, seed, model_name):
    """
    Saves a comparison report as GitHub-Flavored Markdown.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Entropic Sampling: Comparative Benchmark\n\n")
        f.write(f"- **Model:** `{model_name}`\n")
        f.write(f"- **Prompt:** `{prompt}`\n")
        f.write(f"- **Seed:** `{seed}`\n\n")
        f.write("---\n\n")

        for title, data in results.items():
            f.write(f"### {title}\n\n")
            f.write(f"> *{data['description']}*  \n")
            f.write(f"> **Elapsed Time:** {data['time']:.2f}s\n\n")
            f.write("```text\n")
            f.write(data["text"])
            f.write("\n```\n\n")

    print(f"Comparison report saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Entropic Sampling Comparison Tool: Benchmark Greedy vs Temperature vs Alpha-Wave Decoding"
    )
    parser.add_argument("--model_path", type=str, help="Path to local GGUF file or HuggingFace directory")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a futuristic city,",
        help="Input prompt for generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of tokens to generate per method")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for standard sampling baseline (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for standard sampling baseline (default: 0.95)")
    parser.add_argument("--cand_k", type=int, default=12, help="Candidate pool k for future-entropy (default: 12)")
    parser.add_argument("--top_n", type=int, default=10, help="Future tokens n for entropy calculation (default: 10)")
    parser.add_argument("--wavelength", type=float, default=12.0, help="Wavelength of alpha wave (default: 12.0)")
    parser.add_argument("--amp", type=float, default=1.0, help="Amplitude of alpha wave (default: 1.0)")
    parser.add_argument("--instruct", action="store_true", help="Apply model chat template")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Layers to offload to GPU (-1 for all)")
    parser.add_argument("--save_markdown", type=str, default=None, help="Path to write Markdown comparison report")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        model_path = os.environ.get("MODEL_PATH")

    if not model_path:
        print("Scanning local caches for available models...")
        models = list_local_models()
        if not models:
            print("Error: No models found. Specify a path with --model_path.")
            sys.exit(1)

        print("\nAvailable models:")
        for i, m in enumerate(models):
            print(f"[{i + 1}] {m['name']}")

        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    model_path = models[choice_idx]["path"]
                    print(f"Selected: {models[choice_idx]['name']}")
                    break
                print("Invalid selection.")
            except (ValueError, KeyboardInterrupt, EOFError):
                if choice.lower() in ["q", "quit", "exit"]:
                    sys.exit(0)
                print("Please enter a valid number or 'q' to quit.")

    prompt = args.prompt
    instruct = args.instruct
    if not instruct and any(sig in model_path.lower() for sig in ["-it", "instruct", "chat"]):
        if prompt.lower().startswith(("write", "tell", "explain", "describe", "create", "how", "what", "why")):
            instruct = True

    print(f"\nTarget Model: {model_path}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Seed: {args.seed}")
    print(f"Max New Tokens: {args.max_new_tokens}\n")
    print("Beginning comparative sampling runs...\n")

    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        from llama_cpp import Llama

        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
            logits_all=True,
            verbose=False,
        )

        results = run_comparison_llama(
            llm,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cand_k=args.cand_k,
            top_n_future=args.top_n,
            wavelength=args.wavelength,
            amp=args.amp,
            seed=args.seed,
            instruct=instruct,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=device_map,
            torch_dtype=dtype,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = run_comparison_transformers(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cand_k=args.cand_k,
            top_n_future=args.top_n,
            wavelength=args.wavelength,
            amp=args.amp,
            seed=args.seed,
        )

    print_comparison(results, prompt, args.seed)

    if args.save_markdown:
        save_markdown_report(args.save_markdown, results, prompt, args.seed, os.path.basename(model_path))


if __name__ == "__main__":
    main()
