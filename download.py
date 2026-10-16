from huggingface_hub import snapshot_download

from models_pipeline.models_load import MODELS

if __name__ == '__main__':
    for model_id in MODELS:
        print(snapshot_download(model_id), flush=True)
