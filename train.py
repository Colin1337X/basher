import sys
import getpass
import torch

from huggingface_hub import login
from datasets import load_dataset

# -------------------------
# helpers
# -------------------------

def ask(prompt, default=None):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def ask_required(prompt):
    while True:
        val = input(f"{prompt}: ").strip()
        if val:
            return val
        print("Required.")

def ask_int(prompt, default):
    return int(ask(prompt, str(default)))

def ask_float(prompt, default):
    return float(ask(prompt, str(default)))

def header(x):
    print("\n" + "="*60)
    print(x)
    print("="*60 + "\n")

# -------------------------
# main
# -------------------------

header("🔥 Base Model Trainer (CPT / SFT)")

token = getpass.getpass("HF Token: ")
login(token=token)

mode = ask("Mode (CPT/SFT)", "SFT").upper()

model_repo = ask_required("Base model repo")
dataset_repo = ask_required("Dataset repo")
output_repo = ask_required("Output repo")

split = ask("Dataset split", "train")

max_len = ask_int("Max length", 2048)
epochs = ask_float("Epochs", 1 if mode == "CPT" else 2)
lr = ask_float("Learning rate", 1e-5 if mode == "CPT" else 2e-5)
batch = ask_int("Batch size", 2)
grad_accum = ask_int("Grad accum", 8)

# -------------------------
# load dataset
# -------------------------

header("📦 Loading dataset")
dataset = load_dataset(dataset_repo, split=split)
print("Columns:", dataset.column_names)

# =========================
# CPT MODE
# =========================

if mode == "CPT":

    text_col = ask("Text column", "text")

    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != text_col]
    )

    def clean(x):
        t = x.get(text_col, "")
        return {"text": "" if t is None else str(t).strip()}

    dataset = dataset.map(clean)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling
    )

    header("🧠 Loading model")

    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len
        )

    dataset = dataset.map(tokenize, batched=True, num_proc=4)

    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    args = TrainingArguments(
        output_dir="./cpt",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        push_to_hub=True,
        hub_model_id=output_repo
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )

    header("🚀 Training CPT")
    trainer.train()

    model.push_to_hub(output_repo)
    tokenizer.push_to_hub(output_repo)

# =========================
# SFT MODE (JSONL)
# =========================

elif mode == "SFT":

    user_col = ask("User column", "user")
    assistant_col = ask("Assistant column", "assistant")

    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in [user_col, assistant_col]]
    )

    def format_chat(x):
        u = "" if x[user_col] is None else str(x[user_col]).strip()
        a = "" if x[assistant_col] is None else str(x[assistant_col]).strip()
        return {"text": f"User: {u}\nAssistant: {a}"}

    dataset = dataset.map(format_chat)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    header("🧠 Loading Unsloth model")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_repo,
        max_seq_length=max_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )

    args = TrainingArguments(
        output_dir="./sft",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=100,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        push_to_hub=True,
        hub_model_id=output_repo
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_len,
        args=args
    )

    header("🚀 Training SFT")
    trainer.train()

    model.push_to_hub(output_repo)
    tokenizer.push_to_hub(output_repo)

else:
    print("Invalid mode")
    sys.exit(1)

header("✅ DONE")
