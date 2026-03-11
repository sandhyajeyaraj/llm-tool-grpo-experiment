# ============================================================
#  EXPERIMENT — STEP 1: SFT Training
#  Model  : Qwen2.5-1.5B-Instruct
#  Dataset: glaive-function-calling-v2 (real HuggingFace data)
# ============================================================
# !pip install -q transformers datasets peft trl accelerate bitsandbytes

from huggingface_hub import login
login(token="TOKEN")   # replace with your HF token

import json, os, re, torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR  = "/content/qwen15b-glaive-sft"
NUM_SAMPLES = 500       # how many glaive examples to use (max ~113k)
MAX_SEQ_LEN = 512
LORA_RANK   = 16
LORA_ALPHA  = 32
EPOCHS      = 2
BATCH_SIZE  = 2
GRAD_ACCUM  = 8
LR          = 2e-4

# ── LOAD & PARSE GLAIVE DATASET ──────────────────────────────
print("Loading glaive-function-calling-v2 dataset ...")
raw = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
print(f"Total examples in dataset: {len(raw)}")

def parse_glaive_example(row):
    """
    Glaive format:
      system: "SYSTEM: You are helpful... FUNCTION: {json}"
      chat:   "USER: ... ASSISTANT: <functioncall> {...} FUNCTION RESPONSE: {...} ASSISTANT: ..."
    We convert this to standard messages list.
    """
    try:
        system_raw = row.get("system", "")
        chat_raw   = row.get("chat", "")

        # Extract function schema from system field
        func_match = re.search(r'\{.*\}', system_raw, re.DOTALL)
        func_schema = func_match.group(0).strip() if func_match else ""

        # Build system prompt with schema
        system_msg = (
            "You are a helpful assistant with access to tools.\n"
            "When calling a tool respond ONLY with valid JSON:\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
        )
        if func_schema:
            system_msg += f"Available tool:\n{func_schema}"

        messages = [{"role": "system", "content": system_msg}]

        # Parse the chat turns
        # Split on USER:/ASSISTANT:/FUNCTION RESPONSE: markers
        turns = re.split(r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_raw)
        turns = [t.strip() for t in turns if t.strip()]

        i = 0
        while i < len(turns):
            marker = turns[i]
            content = turns[i+1].strip() if i+1 < len(turns) else ""
            # clean <|endoftext|> tokens
            content = content.replace("<|endoftext|>", "").strip()

            if marker == "USER:" and content:
                messages.append({"role": "user", "content": content})
            elif marker == "ASSISTANT:" and content:
                # strip <functioncall> tag, keep the JSON
                content = re.sub(r'<functioncall>\s*', '', content)
                messages.append({"role": "assistant", "content": content})
            elif marker == "FUNCTION RESPONSE:" and content:
                messages.append({"role": "tool", "content": content})
            i += 2

        # Must have at least user + assistant turn
        roles = [m["role"] for m in messages]
        if "user" not in roles or "assistant" not in roles:
            return None

        return messages
    except Exception:
        return None


print("Parsing dataset examples ...")
tokenizer_check = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer_check.pad_token = tokenizer_check.eos_token

texts = []
skipped = 0
for row in raw:
    if len(texts) >= NUM_SAMPLES:
        break
    msgs = parse_glaive_example(row)
    if msgs is None:
        skipped += 1
        continue
    try:
        text = tokenizer_check.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    except Exception:
        skipped += 1

print(f"Parsed {len(texts)} examples  |  Skipped {skipped}")

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(dataset['train'])}  |  Eval: {len(dataset['test'])}")

# ── TOKENIZER ────────────────────────────────────────────────
print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── MODEL ────────────────────────────────────────────────────
print("Loading model (Qwen2.5-1.5B) ...")
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

# ── LORA ─────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ── TRAIN ────────────────────────────────────────────────────
# Auto-detect GPU availability
import torch
has_gpu = torch.cuda.is_available()
print(f"GPU available: {has_gpu}  |  Device: {torch.cuda.get_device_name(0) if has_gpu else 'CPU'}")
if not has_gpu:
    raise RuntimeError(
        "No GPU detected! In Colab go to:\n"
        "  Runtime → Change runtime type → T4 GPU → Save\n"
        "Then reconnect and re-run."
    )

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    seed=42,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=sft_config,
)

print("\n🚀 Starting SFT training ...\n")
trainer.train()

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ SFT adapter saved!")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")
print(f"   Path : {OUTPUT_DIR}")

# ── QUICK INFERENCE ──────────────────────────────────────────
def run_inference(query):
    model.eval()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. When calling a tool respond ONLY with valid JSON: {\"name\": \"<tool>\", \"arguments\": {<args>}}"},
        {"role": "user",   "content": query},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.1,
                             do_sample=True, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("\n-- SFT Inference samples --")
for q in ["Get the weather in Paris", "Calculate 25 * 48", "Search for Python tutorials"]:
    print(f"Q: {q}")
    print(f"A: {run_inference(q)}\n")
