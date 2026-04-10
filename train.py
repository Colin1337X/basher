# =========================
# 🧠 CPT SCRIPT (HF TRAINER)
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

# =========================
# ⚙️ CONFIG
# =========================

MODEL_NAME = "your-moe-model"   # base model
DATASET_NAME = "your-cpt-dataset"

OUTPUT_DIR = "./cpt_model"
MAX_LENGTH = 2048

# =========================
# 📦 LOAD DATASET
# =========================

dataset = load_dataset(DATASET_NAME)

# Expecting:
# {"text": "..."} format

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
    torch_dtype=torch.float16,
    device_map="auto",
)

# =========================
# 🧪 DATA COLLATOR
# =========================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)

# =========================
# ⚙️ TRAINING CONFIG
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=1e-5,
    num_train_epochs=1,   # IMPORTANT: keep small

    logging_steps=50,
    save_steps=1000,

    fp16=True,
    bf16=False,

    report_to="none",
)

# =========================
# 🏋️ TRAIN
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# =========================
# 💾 SAVE
# =========================

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("✅ CPT DONE")