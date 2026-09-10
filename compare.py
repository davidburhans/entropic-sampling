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
    load_llama_model,
)


def standard_temperature_sampling_llama(
    llm,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    seed=42,
    instruct=False,
    skip_thought=True,
    stream=False,
    stream_callback=None,
    return_metrics=False,
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
    for s in ["<turn|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>", "<|endoftext|>", "<|eom_id|>"]:
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

        if chosen_w in stop_tokens:
            break

        generated_tokens.append(chosen_w)
        llm.eval([chosen_w])

        if stream or stream_callback:
            piece = llm.detokenize([chosen_w]).decode("utf-8", errors="replace")
            if callable(stream_callback):
                stream_callback(piece)
            elif stream:
                sys.stdout.write(piece)
                sys.stdout.flush()

    output_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")
    for s in ["<turn|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>"]:
        if output_text.endswith(s):
            output_text = output_text[:-len(s)]

    final_text = output_text.strip() if instruct else (prompt + output_text)
    if return_metrics:
        return final_text, len(generated_tokens)
    return final_text


def standard_temperature_sampling_transformers(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    seed=42,
    stream=False,
    stream_callback=None,
    return_metrics=False,
):
    """
    Standard next-token temperature + top-p sampling with HuggingFace Transformers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

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

            if chosen_w.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, chosen_w], dim=1)

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


def run_comparison_llama(
    llm,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    cand_k=12,
    top_n_future=10,
    wavelength=12.0,
    amp=1.0,
    seed=42,
    instruct=False,
    stream=True,
):
    """
    Runs all 4 sampling methods under the exact same seed and prompt,
    streaming token-by-token output to the terminal in real-time.
    """
    results = {}

    runs = [
        (
            "Greedy (Argmax / Temp 0.0)",
            "[1/4] Greedy Decoding (p(w|c) argmax)",
            "Standard greedy decoding (argmax next-token probability). Frequently funnels into clichés and predictable prose.",
            lambda: standard_temperature_sampling_llama(
                llm, prompt, max_new_tokens=max_new_tokens, temperature=0.0, seed=seed,
                instruct=instruct, stream=stream, return_metrics=True
            ),
        ),
        (
            f"Standard Sampling (Temp {temperature}, Top-p {top_p})",
            f"[2/4] Standard Temperature Sampling (T={temperature}, top-p={top_p})",
            f"Traditional stochastic sampling from the scaled next-token distribution with top-p={top_p}.",
            lambda: standard_temperature_sampling_llama(
                llm, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed,
                instruct=instruct, stream=stream, return_metrics=True
            ),
        ),
        (
            "Static Future-Entropy (alpha = 0.0)",
            "[3/4] Static Future-Entropy (alpha = 0.0 balanced)",
            "Crossfader balance s(w) = p(w|c) * H_hat(w). Weights high future optionality at every token.",
            lambda: future_entropy_sampler_llama(
                llm, prompt, max_new_tokens=max_new_tokens, cand_k=cand_k, top_n_future=top_n_future,
                alpha_constant=0.0, sample=False, instruct=instruct, stream=stream, return_metrics=True
            ),
        ),
        (
            "Alpha-Wave Rhythmic Decoding (Sine Wave)",
            f"[4/4] Alpha-Wave Rhythmic Decoding (wavelength={wavelength}, amp={amp})",
            f"Sinusoidal oscillation of alpha in [-{amp}, +{amp}] with wavelength={wavelength} tokens. Natural breathing cadence between anchors and creative leaps.",
            lambda: future_entropy_sampler_llama(
                llm, prompt, max_new_tokens=max_new_tokens, cand_k=cand_k, top_n_future=top_n_future,
                wavelength=wavelength, amp=amp, sample=False, instruct=instruct, stream=stream, return_metrics=True
            ),
        ),
    ]

    for key, title, desc, run_fn in runs:
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "-"))
        print("=" * 80)
        if stream and not instruct:
            if sys.stdout.isatty():
                sys.stdout.write(f"\033[90m{prompt}\033[0m")
            else:
                sys.stdout.write(prompt)
            sys.stdout.flush()

        t0 = time.time()
        torch.manual_seed(seed)
        random.seed(seed)
        text, n_tokens = run_fn()
        elapsed = time.time() - t0
        tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0.0

        if stream:
            print()
        print(f"\n[Completed: {n_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]")

        results[key] = {
            "text": text,
            "time": elapsed,
            "tokens": n_tokens,
            "speed": tok_per_sec,
            "description": desc,
        }

    return results


def run_comparison_transformers(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    cand_k=12,
    top_n_future=10,
    wavelength=12.0,
    amp=1.0,
    seed=42,
    instruct=False,
    stream=True,
):
    """
    Runs all 4 sampling methods with HuggingFace Transformers,
    streaming token-by-token output to the terminal in real-time.
    """
    if instruct and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt

    results = {}

    runs = [
        (
            "Greedy (Argmax / Temp 0.0)",
            "[1/4] Greedy Decoding (p(w|c) argmax)",
            "Standard greedy decoding (argmax next-token probability).",
            lambda: standard_temperature_sampling_transformers(
                model, tokenizer, formatted_prompt, max_new_tokens=max_new_tokens, temperature=0.0, seed=seed,
                stream=stream, return_metrics=True
            ),
        ),
        (
            f"Standard Sampling (Temp {temperature}, Top-p {top_p})",
            f"[2/4] Standard Temperature Sampling (T={temperature}, top-p={top_p})",
            f"Traditional stochastic sampling with temperature={temperature}.",
            lambda: standard_temperature_sampling_transformers(
                model, tokenizer, formatted_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed,
                stream=stream, return_metrics=True
            ),
        ),
        (
            "Static Future-Entropy (alpha = 0.0)",
            "[3/4] Static Future-Entropy (alpha = 0.0 balanced)",
            "Crossfader balance s(w) = p(w|c) * H_hat(w).",
            lambda: future_entropy_sampler(
                model, tokenizer, formatted_prompt, max_new_tokens=max_new_tokens, cand_k=cand_k, top_n_future=top_n_future,
                alpha_constant=0.0, sample=False, stream=stream, return_metrics=True
            ),
        ),
        (
            "Alpha-Wave Rhythmic Decoding (Sine Wave)",
            f"[4/4] Alpha-Wave Rhythmic Decoding (wavelength={wavelength}, amp={amp})",
            f"Sinusoidal alpha wave (wavelength={wavelength}, amp={amp}).",
            lambda: future_entropy_sampler(
                model, tokenizer, formatted_prompt, max_new_tokens=max_new_tokens, cand_k=cand_k, top_n_future=top_n_future,
                wavelength=wavelength, amp=amp, sample=False, stream=stream, return_metrics=True
            ),
        ),
    ]

    for key, title, desc, run_fn in runs:
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "-"))
        print("=" * 80)
        if stream:
            if sys.stdout.isatty():
                sys.stdout.write(f"\033[90m{prompt}\033[0m")
            else:
                sys.stdout.write(prompt)
            sys.stdout.flush()

        t0 = time.time()
        torch.manual_seed(seed)
        random.seed(seed)
        text, n_tokens = run_fn()
        elapsed = time.time() - t0
        tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0.0

        if stream:
            print()
        print(f"\n[Completed: {n_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]")

        results[key] = {
            "text": text,
            "time": elapsed,
            "tokens": n_tokens,
            "speed": tok_per_sec,
            "description": desc,
        }

    return results


def print_comparison(results, prompt, seed):
    """
    Renders a formatted comparison table and output cards in the terminal.
    """
    width = 80
    sep = "=" * width
    subsep = "-" * width

    print(f"\n{sep}")
    print(" SAMPLING COMPARISON BENCHMARK SUMMARY ".center(width, "#"))
    print(sep)
    print(f"Prompt: {repr(prompt)}")
    print(f"Random Seed: {seed}")
    print(f"{sep}\n")

    # 1. Throughput & Speed Comparison Table
    print(f"{sep}")
    print(" THROUGHPUT & SPEED BENCHMARK ".center(width, "#"))
    print(sep)
    first_key = next(iter(results))
    baseline_speed = results[first_key].get("speed", 1.0)

    header = f"{'Method':<40} {'Tokens':>7} {'Time':>8} {'Speed (tok/s)':>14} {'vs Baseline':>12}"
    print(header)
    print(subsep)
    for title, data in results.items():
        toks = data.get("tokens", 0)
        t = data.get("time", 0.0)
        spd = data.get("speed", 0.0)
        rel = (spd / baseline_speed) if baseline_speed > 0 else 1.0
        rel_str = f"{rel:.2f}x" if title != first_key else "1.00x (base)"
        print(f"{title:<40} {toks:>7d} {t:>7.2f}s {spd:>12.1f} tok/s {rel_str:>12}")
    print(f"{sep}\n")

    # 2. Text outputs
    print(f"{sep}")
    print(" GENERATED TEXT OUTPUTS ".center(width, "#"))
    print(sep)
    for idx, (title, data) in enumerate(results.items(), start=1):
        toks = data.get("tokens", 0)
        t = data.get("time", 0.0)
        spd = data.get("speed", 0.0)
        print(f"\n[{idx}] {title.upper()}")
        print(f"    Rationale: {data['description']}")
        print(f"    Tokens: {toks} | Elapsed Time: {t:.2f}s | Speed: {spd:.1f} tok/s")
        print(subsep)
        print(data["text"])
        print(f"{subsep}")
    print(f"\n{sep}\n")


def save_markdown_report(filepath, results, prompt, seed, model_name):
    """
    Saves a comparison report with speed table as GitHub-Flavored Markdown.
    """
    first_key = next(iter(results))
    baseline_speed = results[first_key].get("speed", 1.0)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Entropic Sampling: Comparative Benchmark\n\n")
        f.write(f"- **Model:** `{model_name}`\n")
        f.write(f"- **Prompt:** `{prompt}`\n")
        f.write(f"- **Seed:** `{seed}`\n\n")
        f.write("---\n\n")

        f.write("## Throughput & Speed Benchmark\n\n")
        f.write("| Sampling Method | Tokens | Generation Time | Speed (tok/s) | Relative Speed |\n")
        f.write("|---|---|---|---|---|\n")
        for title, data in results.items():
            toks = data.get("tokens", 0)
            t = data.get("time", 0.0)
            spd = data.get("speed", 0.0)
            rel = (spd / baseline_speed) if baseline_speed > 0 else 1.0
            rel_str = f"{rel:.2f}x" if title != first_key else "1.00x (baseline)"
            f.write(f"| **{title}** | {toks} | {t:.2f}s | **{spd:.1f} tok/s** | {rel_str} |\n")
        f.write("\n---\n\n")

        f.write("## Detailed Generated Outputs\n\n")
        for idx, (title, data) in enumerate(results.items(), start=1):
            toks = data.get("tokens", 0)
            t = data.get("time", 0.0)
            spd = data.get("speed", 0.0)
            f.write(f"### {idx}. {title}\n\n")
            f.write(f"> *{data['description']}*  \n")
            f.write(f"> **Tokens:** {toks} | **Generation Time:** {t:.2f}s | **Speed:** {spd:.1f} tok/s\n\n")
            f.write("```text\n")
            f.write(data["text"])
            f.write("\n```\n\n")

    print(f"Comparison report saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Entropic Sampling Comparison Tool: Benchmark Greedy vs Temperature vs Alpha-Wave Decoding"
    )
    parser.add_argument("--model_path", "--model", dest="model_path", type=str, help="Path to local GGUF file or HuggingFace directory")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a futuristic city,",
        help="Input prompt for generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=128, help="Number of tokens to generate per method (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for standard sampling baseline (default: 0.8)")
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float, default=0.95, help="Top-p for standard sampling baseline (default: 0.95)")
    parser.add_argument("--cand_k", "--cand-k", dest="cand_k", type=int, default=12, help="Candidate pool k for future-entropy (default: 12)")
    parser.add_argument("--top_n", "--top-n", dest="top_n", type=int, default=10, help="Future tokens n for entropy calculation (default: 10)")
    parser.add_argument("--wavelength", type=float, default=12.0, help="Wavelength of alpha wave (default: 12.0)")
    parser.add_argument("--amp", type=float, default=1.0, help="Amplitude of alpha wave (default: 1.0)")
    parser.add_argument("--instruct", action="store_true", help="Apply model chat template")
    parser.add_argument("--raw", "--no-instruct", dest="raw", action="store_true", help="Force raw prompt completion without chat template")
    parser.add_argument("--no_stream", "--no-stream", dest="no_stream", action="store_true", help="Disable real-time token streaming during comparison runs")
    parser.add_argument("--n_gpu_layers", "--n-gpu-layers", dest="n_gpu_layers", type=int, default=-1, help="Layers to offload to GPU (-1 for all)")
    parser.add_argument("--save_markdown", "--save-markdown", dest="save_markdown", type=str, default=None, help="Path to write Markdown comparison report")
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
    if not args.raw and not instruct:
        is_it_model = any(sig in model_path.lower() for sig in ["-it", "instruct", "chat"])
        is_gemma = "gemma" in model_path.lower()
        if is_gemma or is_it_model:
            instruct = True
            print("Auto-detected instruct-tuned model. Enabling chat template formatting (pass --raw to disable).")

    stream = not args.no_stream

    print(f"\nTarget Model: {model_path}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Seed: {args.seed}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Streaming: {'Enabled' if stream else 'Disabled'}\n")
    print("Beginning comparative sampling runs...\n")

    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        llm = load_llama_model(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
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
            stream=stream,
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
            instruct=instruct,
            stream=stream,
        )

    print_comparison(results, prompt, args.seed)

    if args.save_markdown:
        save_markdown_report(args.save_markdown, results, prompt, args.seed, os.path.basename(model_path))


if __name__ == "__main__":
    main()
