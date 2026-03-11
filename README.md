# 🛠️ LLM Tool Call Fine-Tuning — SFT + GRPO

Fine-tuning **Qwen 2.5-1.5B** to make structured JSON tool calls using SFT and Reinforcement Learning (GRPO).

---

## 🧪 Experiment

Train a small LLM to respond with structured tool calls instead of plain text:

```json
{"name": "get_weather", "arguments": {"location": "Paris"}}
```

**Three steps:**
1. **SFT** — teach the model using 500 real examples from glaive-function-calling-v2
2. **GRPO** — improve using reward functions (no labeled answers needed)
3. **Eval** — compare SFT vs GRPO side by side on 12 test queries

---

## 📊 Results

| Metric | SFT | GRPO | Winner |
|--------|-----|------|--------|
| JSON Valid | 0% | 92% | GRPO ✅ |
| Correct Tool | 0% | 50% | GRPO ✅ |
| Has Arguments | 0% | 42% | GRPO ✅ |
| Clean Output | 0% | 92% | GRPO ✅ |
| Avg Quality Score | 0.0 | 0.59 | GRPO ✅ |

> **Key finding:** SFT model answered questions directly in plain text (never used tools).
> GRPO model learned to always respond with structured JSON tool calls.
> Tested on 12 queries across weather, calculator, search, stocks, translation and unit conversion.

---

## 🚀 Run on Colab

```bash
!pip install -q transformers datasets peft trl accelerate bitsandbytes
!python EXP_STEP1_sft.py    # ~20 min
!python EXP_STEP2_grpo.py   # ~30 min
!python EXP_STEP3_compare.py
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `EXP_STEP1_sft.py` | SFT training (Colab) |
| `EXP_STEP2_grpo.py` | GRPO RL training (Colab) |
| `EXP_STEP3_compare.py` | Evaluation + comparison |

---

## 🔧 Stack

`Qwen2.5-1.5B` • `QLoRA` • `GRPO` • `glaive-function-calling-v2` • `trl` • `peft`

---

## 📎 References

- [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [GRPO — DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)