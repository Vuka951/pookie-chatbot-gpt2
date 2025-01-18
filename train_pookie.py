import os
import shutil
import torch
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset

# Without this there is a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1) Configuration
MODEL_DIR = "./pookie-chatbot"
DATASET_FOLDER = "./datasets"        # folder containing .txt files
OUTPUT_DIR = "./pookie-chatbot"      # output directory for fine-tuned model, using same in order to override it

BATCH_SIZE = 2           # keeping small to not run out of memory
EPOCHS = 10              # Needs to be high with small datasets
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5
BLOCK_SIZE = 512         # Needs to be this or even samller to ot run out of memory

# Check for MPS (Metal) on Mac, else CPU
use_mps = torch.backends.mps.is_available()
device_str = "mps" if use_mps else "cpu"

# Use CUDA is available
use_cuda = torch.cuda.is_available()

# Decide on fp16 or bf16
fp16_flag = False
bf16_flag = False

if use_cuda:
    # Standard GPU, use fp16
    fp16_flag = True
elif use_mps:
    # MPS may allow bf16 in some PyTorch versions
    bf16_flag = True
else:
    # CPU only
    fp16_flag = False
    bf16_flag = False

# ------------------------------------------------------------------------------
# 2) Custom Dataset
# ------------------------------------------------------------------------------
class PookieTextDataset(Dataset):
    def __init__(self, tokenizer, folder_path, block_size=512):
        self.examples = []
        
        txt_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith(".txt")
        ]
        if not txt_files:
            raise ValueError(f"No .txt files found in {folder_path}")
        
        # Read & combine
        all_texts = []
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                all_texts.append(text)
        full_text = "\n".join(all_texts)
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False
        )["input_ids"][0]  # shape: (seq_len,)

        # Split into chunks
        for i in range(0, len(tokenized), block_size):
            chunk = tokenized[i : i + block_size]
            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    print("=== Starting Pookie GPT-2 Fine-Tuning ===")
    print(f"Device: {device_str}")
    
    # 3) Load tokenizer & model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    
    # 4) Create dataset
    train_dataset = PookieTextDataset(
        tokenizer=tokenizer,
        folder_path=DATASET_FOLDER,
        block_size=BLOCK_SIZE
    )
    
    # 5) Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="epoch",          # save once per epoch
        logging_strategy="epoch",       # log once per epoch
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="no",       # or "epoch" if you have eval data
        do_train=True,
        do_eval=False,
        
        # Mixed precision flags
        fp16=fp16_flag,   # use FP16 on CUDA
        bf16=bf16_flag,   # use BF16 on MPS (if available)
        
        # Gradient checkpointing can reduce memory usage
        gradient_checkpointing=True,
        
        # gradient_accumulation_steps=2, # same as batch size = BATCH_SIZE*2
    )
    
    # 7) Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 8) Train
    trainer.train()
    
    # 9) Save model & tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model & tokenizer saved to: {OUTPUT_DIR}")

    # 10) Remove checkpoints to save space
    for item in os.listdir(OUTPUT_DIR):
        if item.startswith("checkpoint-"):
            checkpoint_dir = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(checkpoint_dir):
                print(f"Removing checkpoint directory: {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)

    print("All checkpoint directories removed.")

if __name__ == "__main__":
    main()
