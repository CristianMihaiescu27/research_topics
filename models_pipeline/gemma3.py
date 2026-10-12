from vllm import LLM, SamplingParams


class Gemma3():
    def __init__(self):
        pass
    
    def download_and_load_model(self, save_folder=None):
        print(f"Downloading/Loading RedHatAI/gemma-3n-E4B-it-FP8-dynamic...")

        model = LLM(
            model="RedHatAI/gemma-3n-E4B-it-FP8-dynamic", 
            max_model_len=4096
        )

        return model


    def generate_response(self, model, prompt):
        # model's expected chat structure
        prompt = [{"role": "user", "content": prompt}] # prompt
        sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

        outputs = model.chat(
            messages=prompt, 
            sampling_params=sampling_params, 
            use_tqdm=False
        )
        
        # Return the text
        return outputs[0].outputs[0].text.strip()
