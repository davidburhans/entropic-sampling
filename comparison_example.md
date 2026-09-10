# Entropic Sampling: Comparative Benchmark

- **Model:** `gemma-4-E4B_q4_0-it.gguf`
- **Prompt:** `Deep in the enchanted forest, an ancient key`
- **Seed:** `123`

---

### Greedy (Argmax / Temp 0.0)

> *Standard greedy decoding (argmax next-token probability). Frequently funnels into clichés and predictable prose.*  
> **Elapsed Time:** 0.15s

```text
Deep in the enchanted forest, an ancient key lies hidden. To find it, one must solve the riddle of the forest.

**The Riddle:**

*I have
```

### Standard Sampling (Temp 0.8, Top-p 0.95)

> *Traditional stochastic sampling from the scaled next-token distribution with top-p=0.95.*  
> **Elapsed Time:** 0.58s

```text
Deep in the enchanted forest, an ancient key rests upon a moss-covered stone. This key holds the power to unlock the secrets of the forest. A gentle breeze rust
```

### Static Future-Entropy (alpha = 0.0)

> *Crossfader balance s(w) = p(w|c) * H_hat(w). Weights high future optionality at every token.*  
> **Elapsed Time:** 0.85s

```text
Deep in the enchanted forest, an ancient key, whispered secrets of forgotten realms. It was said that only those with a heart of pure gold could unlock its mysteries.


```

### Alpha-Wave Rhythmic Decoding (Sine Wave)

> *Sinusoidal oscillation of alpha in [-1.0, +1.0] with wavelength=12.0 tokens. Natural breathing cadence between anchors and creative leaps.*  
> **Elapsed Time:** 0.92s

```text
Deep in the enchanted forest, an ancient key, whispered of in the dark, lay hidden. It was said that only true seekers, those with a pure heart, could
```

