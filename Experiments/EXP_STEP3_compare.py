# ============================================================
#  EXPERIMENT — STEP 3: Compare SFT vs GRPO
#  Evaluates both adapters on same test set and prints
#  a side-by-side comparison table
# ============================================================
# !pip install -q transformers peft accelerate bitsandbytes

from huggingface_hub import login
login(token="TOKEN")   # replace with your HF token

import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_PATH    = "/content/qwen15b-glaive-sft"
GRPO_PATH   = "/content/qwen15b-glaive-grpo"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools.\n"
    "When calling a tool respond ONLY with valid JSON:\n"
    '{"name": "<tool_name>", "arguments": {<args>}}\n'
    "Do not add any explanation — just the JSON."
)

# ── TEST CASES ───────────────────────────────────────────────
# Mix of in-distribution and out-of-distribution queries
TEST_CASES = [
    # Weather
    {"query": "What's the weather in Tokyo?",
     "expected_tool": "get_current_weather",
     "expected_args": ["location"]},
    {"query": "Is it raining in London right now?",
     "expected_tool": "get_current_weather",
     "expected_args": ["location"]},
    {"query": "Tell me the temperature in Berlin.",
     "expected_tool": "get_current_weather",
     "expected_args": ["location"]},

    # Calculator / Math
    {"query": "What is 15% of 840?",
     "expected_tool": "calculate",
     "expected_args": ["expression"]},
    {"query": "Calculate the square root of 256.",
     "expected_tool": "calculate",
     "expected_args": ["expression"]},
    {"query": "How much is 128 divided by 4 plus 56 times 3?",
     "expected_tool": "calculate",
     "expected_args": ["expression"]},

    # Search
    {"query": "Search for the latest news on AI.",
     "expected_tool": "search",
     "expected_args": ["query"]},
    {"query": "Find Python tutorials for beginners.",
     "expected_tool": "search",
     "expected_args": ["query"]},

    # Stock / Finance
    {"query": "What is the stock price of Apple?",
     "expected_tool": "get_stock_price",
     "expected_args": ["symbol"]},
    {"query": "Get me the current price of Tesla shares.",
     "expected_tool": "get_stock_price",
     "expected_args": ["symbol"]},

    # Translation
    {"query": "Translate 'Hello, how are you?' to French.",
     "expected_tool": "translate",
     "expected_args": ["text", "target_language"]},

    # Unit conversion
    {"query": "Convert 100 kilometers to miles.",
     "expected_tool": "convert_units",
     "expected_args": ["value", "from_unit", "to_unit"]},
]

# ── SCORING ──────────────────────────────────────────────────
def parse_json(text):
    text = text.strip()
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None

def score(output_text, expected_tool, expected_args):
    parsed = parse_json(output_text)
    if parsed is None:
        return {"json": False, "tool": False, "args": False,
                "clean": False, "score": 0.0, "parsed": None}

    # Tool name — allow partial match (model may use slightly different name)
    tool_out  = str(parsed.get("name", "")).lower()
    tool_exp  = expected_tool.lower()
    tool_ok   = (tool_out == tool_exp) or (tool_exp in tool_out) or (tool_out in tool_exp)

    # Args — check expected keys exist in output arguments
    args_out  = parsed.get("arguments", {})
    args_ok   = all(
        any(exp_k in out_k or out_k in exp_k
            for out_k in str(args_out).lower().split())
        for exp_k in expected_args
    ) if isinstance(args_out, dict) and args_out else False

    # Clean output (no extra text)
    start = output_text.find("{")
    end   = output_text.rfind("}") + 1
    extra = len(output_text[:start].strip()) + len(output_text[end:].strip())
    clean = extra < 15

    sc = (0.3 * tool_ok) + (0.4 * args_ok) + (0.3 * clean)

    return {"json": True, "tool": tool_ok, "args": args_ok,
            "clean": clean, "score": round(sc, 2), "parsed": parsed}

# ── LOAD MODEL HELPER ────────────────────────────────────────
def load_model_with_adapter(adapter_path, label):
    print(f"\nLoading {label} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print(f"  {label} ready.")
    return model, tokenizer

def infer(model, tokenizer, query):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, temperature=0.1,
            do_sample=True, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── RUN COMPARISON ───────────────────────────────────────────
sft_model,  sft_tok  = load_model_with_adapter(SFT_PATH,  "SFT model")
grpo_model, grpo_tok = load_model_with_adapter(GRPO_PATH, "GRPO model")

sft_results  = []
grpo_results = []

print("\n" + "=" * 80)
print(f"{'SIDE-BY-SIDE COMPARISON: SFT vs GRPO (RL)':^80}")
print("=" * 80)

for i, tc in enumerate(TEST_CASES):
    sft_out  = infer(sft_model,  sft_tok,  tc["query"])
    grpo_out = infer(grpo_model, grpo_tok, tc["query"])

    sft_s  = score(sft_out,  tc["expected_tool"], tc["expected_args"])
    grpo_s = score(grpo_out, tc["expected_tool"], tc["expected_args"])

    sft_results.append(sft_s)
    grpo_results.append(grpo_s)

    winner = "GRPO" if grpo_s["score"] > sft_s["score"] else ("SFT" if sft_s["score"] > grpo_s["score"] else "TIE")

    print(f"\n[{i+1:02d}] {tc['query']}")
    print(f"  Expected tool : {tc['expected_tool']} | args: {tc['expected_args']}")
    print(f"  SFT  [{sft_s['score']:.2f}] : {sft_out[:100]}")
    print(f"  GRPO [{grpo_s['score']:.2f}] : {grpo_out[:100]}")
    print(f"  Winner → {winner}")

# ── SUMMARY TABLE ────────────────────────────────────────────
def summarise(results, label):
    n = len(results)
    return {
        "label":      label,
        "json":       sum(r["json"]  for r in results),
        "tool":       sum(r["tool"]  for r in results),
        "args":       sum(r["args"]  for r in results),
        "clean":      sum(r["clean"] for r in results),
        "avg_score":  round(sum(r["score"] for r in results) / n, 3),
        "n":          n,
    }

sft_sum  = summarise(sft_results,  "SFT")
grpo_sum = summarise(grpo_results, "GRPO (RL)")

print("\n" + "=" * 80)
print(f"{'FINAL SUMMARY':^80}")
print("=" * 80)
print(f"{'Metric':<25} {'SFT':>15} {'GRPO (RL)':>15} {'Winner':>12}")
print("-" * 80)

metrics = [
    ("JSON Valid",      "json"),
    ("Correct Tool",    "tool"),
    ("Has Arguments",   "args"),
    ("Clean Output",    "clean"),
]
for label, key in metrics:
    sv = sft_sum[key];  gv = grpo_sum[key];  n = sft_sum["n"]
    sp = f"{sv}/{n} ({100*sv/n:.0f}%)"
    gp = f"{gv}/{n} ({100*gv/n:.0f}%)"
    w  = "GRPO ✅" if gv > sv else ("SFT ✅" if sv > gv else "TIE")
    print(f"  {label:<23} {sp:>15} {gp:>15} {w:>12}")

print("-" * 80)
sp = str(sft_sum["avg_score"]);  gp = str(grpo_sum["avg_score"])
w  = "GRPO ✅" if grpo_sum["avg_score"] > sft_sum["avg_score"] else ("SFT ✅" if sft_sum["avg_score"] > grpo_sum["avg_score"] else "TIE")
print(f"  {'Avg Quality Score':<23} {sp:>15} {gp:>15} {w:>12}")
print("=" * 80)

print("\n📊 Interpretation:")
print("  JSON Valid   → Basic format learned")
print("  Correct Tool → Model picks the right function")
print("  Has Arguments→ Model provides required parameters")
print("  Clean Output → No extra text outside JSON")
print("  Avg Score    → Combined weighted quality (0.0 – 1.0)")
print("\n  GRPO should show improvement especially in:")
print("  → Cleaner output (no extra text)")
print("  → More consistent argument structure")
print("  → Better generalisation to unseen tool types")
