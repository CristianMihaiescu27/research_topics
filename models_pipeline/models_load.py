from models_pipeline.llama3 import Llama3
from gemma3 import Gemma3
from llama2 import Llama2

class Model_controler():
    def __init__(self, model_name):
        self.model_name = model_name

        self.models = {"LLaMAX/LLaMAX3-8B-Alpaca": Llama3(), "RedHatAI/gemma-3n-E4B-it-FP8-dynamic": Gemma3(),
                       "LLaMAX/LLaMAX2-7B": Llama2}

        self.selected_model = self.models[self.model_name]

        pass

    
    def download_model(self):
        # Download/Load the model into a specific folder
        self.model, self.tokenizer = self.download_and_load_model(
            model_id=self.model_name, 
            save_folder="./my_local_models"
        )
        


    def pipeline(self, prompt):
        # Prompt Example
        prompt = "Write a short poem about a server rack. Make it about Dexter series."
        response = self.selected_model.generate_response(self.model, prompt, self.tokenizer)
        print(response)



name = "LLaMAX/LLaMAX3-8B-Alpaca"
controler = Model_controler(name)
controler.pipeline()
