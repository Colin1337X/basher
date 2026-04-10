# =========================
# 🧠 CPT SCRIPT (HF + CSV TEXT ONLY)
# =========================

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login

# =========================
# 🔑 LOGIN
# =========================

login()  # or login("hf_xxx")

# =========================
# ⚙️ CONFIG
# =========================

MODEL_NAME = "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B"
DATASET_NAME = "Colin1337X/ZoinkedRP"

OUTPUT_REPO = "Colin1337X/zoinkedrp-cpt"

MAX_LENGTH = 2048

# =========================
# 📦 LOAD DATASET
# =========================

dataset = load_dataset(DATASET_NAME, split="train")

# Keep ONLY text column
dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col != "text"]
)

# =========================
# 🧹 CLEAN DATA
# =========================

def clean(example):
    text = example["text"]
    if text is None:
        return {"text": ""}
    return {"text": text.strip()}

dataset = dataset.map(clean)

# Remove empty rows
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

# =========================
# 🔤 TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize, batched=True, num_proc=4)

# =========================
# 🧠 MODEL
# =========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # H100 sweet spot
    device_map="auto",
)

# =========================
# 📚 DATA COLLATOR
# =========================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# =========================
# ⚙️ TRAINING CONFIG
# =========================

training_args = TrainingArguments(
    output_dir="./cpt",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=1e-5,
    num_train_epochs=1,   # DO NOT increase

    logging_steps=50,
    save_steps=1000,

    bf16=True,   # faster on H100
    fp16=False,

    report_to="none",

    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
)

# =========================
# 🏋️ TRAIN
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# =========================
# 💾 SAVE + PUSH
# =========================

model.push_to_hub(OUTPUT_REPO)
tokenizer.push_to_hub(OUTPUT_REPO)

print("✅ CPT COMPLETE + UPLOADED")