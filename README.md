# Entropic Sampling: Future-Entropy Sampler & Alpha-Wave Decoding

An implementation of Count Bayesie's (Will Kurt's) **Future-Entropy Sampler** with **Alpha-Wave Rhythmic Decoding** for creative text generation with local LLMs.

Based on: [Making LLMs Better at Creative Writing using Entropy](https://www.countbayesie.com/blog/2026/7/1/making-llms-better-at-creative-writing-using-entropy).

---

## The Concept

Traditional LLM samplers (greedy, temperature, top-p/top-k) only consider the distribution of the immediate next token $p(v \mid c)$. By nature, models tend to drift towards the predictable average, often leading to dull or formulaic prose.

The **Future-Entropy Sampler** looks one step ahead:
1. For each top-$k$ candidate token $w$, we evaluate the model's conditional distribution over the token that would follow:
   $$q_w(V) = p(V \mid c, w)$$
2. We compute the normalized Shannon entropy $\hat{H}(w) \in [0, 1]$ over the top-$n$ future tokens:
   $$\hat{H}(w) = \frac{-\sum_{v \in T_n(w)} \tilde{q}_w(v) \log \tilde{q}_w(v)}{\log n}$$
   High entropy means choosing $w$ opens up diverse, creative continuations; low entropy means $w$ funnels into a cliché or forced continuation.
3. We score each candidate using the $\alpha$ crossfader:
   $$s(w) = p(w \mid c)^a \cdot \hat{H}(w)^b$$
   where:
   $$a = 1 - \max(0, \alpha), \quad b = 1 - \max(0, -\alpha), \quad \alpha \in [-1, 1]$$
   - $\alpha = -1.0$: pure probability (greedy/predictable)
   - $\alpha = 0.0$: balanced future-entropy ($p \cdot \hat{H}$)
   - $\alpha = +1.0$: pure future-entropy (maximum optionality/surprise)
4. We choose $w = \arg\max s(w)$ from the candidate pool.
5. In **$\alpha$-wave rhythmic decoding**, $\alpha$ oscillates smoothly along a sine wave:
   $$\alpha(\text{step}) = \text{amp} \cdot \sin\left(\frac{2\pi \cdot \text{step}}{\text{wavelength}}\right)$$
   allowing text to naturally breathe between stabilizing anchors and inventive, surprising flourishes.

---

## Quickstart

### Interactive Model Selection
```bash
uv run python sampler.py
```
Scans your local caches (HuggingFace Hub, LM Studio, GPT4All) for compatible models and presents an interactive menu.

### Direct Command Line Run
```bash
# Creative continuation with Gemma 4 or Qwen GGUF
uv run python sampler.py \
  --model_path /path/to/model.gguf \
  --prompt "Once upon a time in a futuristic city," \
  --max_new_tokens 80 \
  --cand_k 12 \
  --top_n 10 \
  --wavelength 12.0 \
  --amp 1.0 \
  --verbose_steps
```

### Instruct Mode
For instruction-tuned models with a specific task:
```bash
uv run python sampler.py \
  --model_path /path/to/model.gguf \
  --prompt "Write the opening of a ghost story set in an old lighthouse." \
  --instruct \
  --max_new_tokens 100
```

---

## CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_path` | `str` | `None` | Path to GGUF file or HuggingFace directory |
| `--prompt` | `str` | `"Once upon a time..."` | Input prompt or story opening |
| `--instruct` | `flag` | `False` | Apply model chat template with thought channel bypass |
| `--max_new_tokens` | `int` | `80` | Number of tokens to generate |
| `--cand_k` | `int` | `12` | Number of candidates $k$ to consider at each step |
| `--top_n` | `int` | `10` | Top $n$ future tokens for Shannon entropy |
| `--wavelength` | `float` | `12.0` | Alpha sine wave wavelength in tokens |
| `--amp` | `float` | `1.0` | Amplitude of the alpha wave $[-1.0, +1.0]$ |
| `--alpha` | `float` | `None` | Static alpha in $[-1.0, 1.0]$ to disable wave |
| `--sample` | `flag` | `False` | Use stochastic multinomial sampling instead of argmax |
| `--n_gpu_layers` | `int` | `-1` | GPU layers offloaded to llama.cpp (-1 for all) |
| `--verbose_steps` | `flag` | `False` | Print step-by-step alpha, $a$, $b$, and chosen tokens |
