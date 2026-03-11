# ============================================================
#  EXPERIMENT — STEP 2: GRPO Reinforcement Learning
#  Starts from the SFT adapter and improves with RL rewards
#  Model  : Qwen2.5-1.5B-Instruct + SFT adapter
#  Method : GRPO (Group Relative Policy Optimization)
# ============================================================
# !pip install -q transformers datasets peft trl accelerate bitsandbytes

from huggingface_hub import login
login(token="TOKEN")   # replace with your HF token

import json, os, re, torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER   = "/content/qwen15b-glaive-sft"    # from STEP 1
OUTPUT_DIR    = "/content/qwen15b-glaive-grpo"
NUM_SAMPLES   = 200     # RL needs fewer samples than SFT
MAX_SEQ_LEN   = 512
GRPO_EPOCHS   = 1
BATCH_SIZE    = 2
LR            = 1e-5    # lower LR for RL fine-tuning

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools.\n"
    "When calling a tool respond ONLY with valid JSON:\n"
    '{"name": "<tool_name>", "arguments": {<args>}}\n'
    "Do not add any explanation — just the JSON."
)

# ── REWARD FUNCTIONS ─────────────────────────────────────────
# These are the core of GRPO — the model learns to maximise these scores

def reward_json_valid(completions, **kwargs) -> list[float]:
    """Reward 1: Is the output valid JSON at all?"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        text = text.strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            json.loads(text[start:end])
            rewards.append(1.0)   # valid JSON
        except Exception:
            rewards.append(-1.0)  # invalid / no JSON
    return rewards

def reward_has_name_field(completions, **kwargs) -> list[float]:
    """Reward 2: Does the JSON have a 'name' field (tool name)?"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        text = text.strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            rewards.append(1.0 if "name" in obj else 0.0)
        except Exception:
            rewards.append(-0.5)
    return rewards

def reward_has_arguments_field(completions, **kwargs) -> list[float]:
    """Reward 3: Does the JSON have an 'arguments' field?"""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        text = text.strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            rewards.append(1.0 if "arguments" in obj and isinstance(obj["arguments"], dict) else 0.0)
        except Exception:
            rewards.append(-0.5)
    return rewards

def reward_no_extra_text(completions, **kwargs) -> list[float]:
    """Reward 4: Penalise if model adds text outside the JSON block."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        text = text.strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            before = text[:start].strip()
            after  = text[end:].strip()
            # Penalise if there is significant extra text
            extra = len(before) + len(after)
            rewards.append(1.0 if extra < 10 else max(0.0, 1.0 - extra / 100))
        except Exception:
            rewards.append(-0.5)
    return rewards

def reward_correct_tool_format(completions, prompts=None, **kwargs) -> list[float]:
    """
    Reward 5: Combined quality score.
      +1.0  valid JSON with both name + arguments fields, no extra text
      +0.5  valid JSON with name field only
      +0.2  valid JSON but missing required fields
       0.0  no JSON found
      -1.0  completely invalid output
    """
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        text = text.strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            obj   = json.loads(text[start:end])

            has_name = "name" in obj and isinstance(obj.get("name"), str) and len(obj["name"]) > 0
            has_args = "arguments" in obj and isinstance(obj.get("arguments"), dict)
            extra    = len(text[:start].strip()) + len(text[end:].strip())
            clean    = extra < 15

            if has_name and has_args and clean:
                rewards.append(1.0)
            elif has_name and has_args:
                rewards.append(0.7)
            elif has_name:
                rewards.append(0.5)
            else:
                rewards.append(0.2)
        except (ValueError, json.JSONDecodeError):
            rewards.append(-1.0)
    return rewards

# ── LOAD DATASET FOR RL ──────────────────────────────────────
print("Loading glaive dataset for GRPO ...")
raw = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

# For GRPO we only need the prompts (questions), not the answers
# The reward function grades the model's output
def extract_prompt(row):
    try:
        chat_raw = row.get("chat", "")
        system_raw = row.get("system", "")

        func_match = re.search(r'\{.*?\}', system_raw, re.DOTALL)
        func_schema = func_match.group(0).strip() if func_match else ""

        # Get first USER turn only
        user_match = re.search(r'USER:\s*(.*?)(?=ASSISTANT:|$)', chat_raw, re.DOTALL)
        if not user_match:
            return None

        user_msg = user_match.group(1).replace("<|endoftext|>", "").strip()
        if len(user_msg) < 5:
            return None

        sys_msg = SYSTEM_PROMPT
        if func_schema:
            sys_msg += f"\nAvailable tool:\n{func_schema}"

        return [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ]
    except Exception:
        return None

print("Building GRPO prompts ...")
prompts = []
for row in raw:
    if len(prompts) >= NUM_SAMPLES:
        break
    p = extract_prompt(row)
    if p:
        prompts.append({"prompt": p})

grpo_dataset = Dataset.from_list(prompts)
print(f"GRPO dataset size: {len(grpo_dataset)}")

# ── LOAD MODEL + SFT ADAPTER ─────────────────────────────────
print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # GRPO needs left padding

print("Loading base model ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.config.use_cache = False

# Load the SFT adapter as starting point
print(f"Loading SFT adapter from {SFT_ADAPTER} ...")
model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
print("Model ready for GRPO.\n")

# ── GRPO CONFIG ──────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=GRPO_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    # GRPO specific
    num_generations=4,          # generate 4 outputs per prompt, rank them
    max_completion_length=128,  # renamed from max_new_tokens
    seed=42,
)

# Generation kwargs passed separately
os.environ["GRPO_TEMPERATURE"] = "0.9"

# ── GRPO TRAINER ─────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_json_valid,            # weight: ensures basic JSON
        reward_has_name_field,        # weight: ensures tool name present
        reward_has_arguments_field,   # weight: ensures args present
        reward_no_extra_text,         # weight: keeps output clean
        reward_correct_tool_format,   # weight: combined quality score
    ],
    args=grpo_config,
    train_dataset=grpo_dataset,
)

print("🚀 Starting GRPO (RL) training ...\n")
print("What GRPO does each step:")
print("  1. Generate 4 different tool call candidates per prompt")
print("  2. Score each with reward functions above")
print("  3. Update model to prefer higher-scoring outputs\n")
trainer.train()

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ GRPO adapter saved to {OUTPUT_DIR}")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")
