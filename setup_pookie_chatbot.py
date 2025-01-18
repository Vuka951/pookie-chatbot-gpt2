import os
import torch
import logging
import torch.nn.init as init
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1) Create a custom tokenizer from scratch
# ------------------------------------------------------------------------------
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# <user> and <system> tokens used for formatting and <sep> for multi-turn separation
trainer = BpeTrainer(
    special_tokens=[
        "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<user>", "<system>"
    ]
)

# Collect all .txt files from the "datasets" folder
folder_path = "./datasets"
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".txt")
]
if not files:
    raise ValueError(f"No .txt files found in '{folder_path}'.")

# Train the tokenizer on all files in the folder
logger.info("Tokenizer training started.")
tokenizer.train(files, trainer)
logger.info("Tokenizer training completed.")

# Save the raw tokenizer
tokenizer.save("custom_tokenizer.json")

# ------------------------------------------------------------------------------
# 2) Convert to a PreTrainedTokenizerFast and add special tokens
# ------------------------------------------------------------------------------
tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_tokenizer.json")
tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<sep>", "<user>", "<system>"]
})

# ------------------------------------------------------------------------------
# 3) Configure and initialize the GPT-2 model
# ------------------------------------------------------------------------------
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=2048,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1
)

model = GPT2LMHeadModel(config)

# Adjust model embeddings to match newly added tokens
model.resize_token_embeddings(len(tokenizer))

# ------------------------------------------------------------------------------
# 4) Light initialization of weights
# ------------------------------------------------------------------------------
with torch.no_grad():
    for param in model.parameters():
        if param.dim() > 1:
            init.xavier_uniform_(param)

# ------------------------------------------------------------------------------
# 5) Save the fresh model and tokenizer
# ------------------------------------------------------------------------------
save_directory = './pookie-chatbot'
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

logger.info(f"Pookie chatbot model and tokenizer saved to '{save_directory}'")
