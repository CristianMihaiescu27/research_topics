import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Llama2():
    def __init__(self):
        pass

    def download_and_load_model(self, save_folder=None):
        """
        Downloads and loads the model and tokenizer.
        If save_folder is provided, the weights are cached in that specific directory.
        """
        print(f"Downloading/Loading LLaMAX/LLaMAX2-7B...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id="LLaMAX/LLaMAX2-7B", 
            cache_dir=save_folder
        )

        # Load model 
        # (Using bfloat16 is highly recommended for 8B models to save RAM)
        model = AutoModelForCausalLM.from_pretrained(
            model_id="LLaMAX/LLaMAX2-7B", 
            cache_dir=save_folder,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Automatically utilizes GPU if available
        )

        print("Model loaded successfully!")
        return model, tokenizer


    def generate_response(self, model, prompt, tokenizer):
        """
        Formats the prompt using the Alpaca template, inserts the user's prompt, 
        and generates a response.
        """
        # LLaMAX Alpaca prompt template
        # if input_text:
        #     prompt = (
        #         "Below is an instruction that describes a task, paired with an input that provides further context. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         f"### Instruction:\n{instruction}\n\n"
        #         f"### Input:\n{input_text}\n\n"
        #         "### Response:\n"
        #     )
        # else:
        #     prompt = (
        #         "Below is an instruction that describes a task. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         f"### Instruction:\n{instruction}\n\n"
        #         "### Response:\n"
        #     )
            
        # Tokenize input and move to the same device as the model (e.g., GPU)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,       # Adjust for longer/shorter responses
                temperature=0.6,          # Lower = more precise, Higher = more creative
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated tokens back to text
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Isolate and return just the generated response text
        response = full_response.split("### Response:\n")[-1].strip()
        return response
