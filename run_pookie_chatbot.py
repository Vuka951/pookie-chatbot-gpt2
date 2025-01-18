import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

MODEL_DIR = "./pookie-chatbot"
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_response(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.4
):
    """
    Generate a text response given the input prompt.
    Uses max_new_tokens so the total length can exceed that of the prompt.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    # skip_special_tokens=False, strip is done manually below
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return generated_text

def main():
    print("=== Welcome to the Pookie Chatbot! ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Convos stored in a list, each turn is <user> ... or <system> ...
    turns = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat. Goodbye Pookie!")
            break
        
        # 1) Add the user's turn with <user>
        turns.append(f"<user> {user_input.strip()}")
        
        # 2) Build short conversation, last 5 pairs => last 10 lines
        short_history = turns[-10:]
        
        # 3) Create the prompt, starting with <bos>, then each turn separated by <sep>,
        #    and end with <sep> <system> to cue the model to produce system's text next.
        # max length of the hitory is 700
        prompt_for_model = "<bos>" + " <sep> ".join(short_history)[:700] + " <sep> <system> "
        
        # 4) Generate the model output
        raw_output = generate_response(prompt_for_model)

        # 5) Extract the newly generated system text
        system_prefix = "<sep> <system> "
        if system_prefix in raw_output:
            # Split from the last occurrence
            parts = raw_output.rsplit(system_prefix, 1)
            system_text = parts[-1]
        else:
            system_text = raw_output

        # Cut off at <eos> if present
        if "<eos>" in system_text:
            system_text = system_text.split("<eos>", 1)[0]

        # Remove leftover special tokens
        for token in ["<bos>", "<sep>", "<user>", "<system>", "<eos>"]:
            system_text = system_text.replace(token, "")
        system_text = system_text.strip()
        
        # Print the chatbot response
        print(f"Pookie Chatbot: {system_text}")

        # 6) Append system turn to the conversation
        turns.append(f"<system> {system_text}")

if __name__ == "__main__":
    main()
