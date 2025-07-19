from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

MODEL_CACHE = {}
MODEL_REPO = "Charifa/djos-all-models"
SUBDIR = "stt"

def load_stt_model():
    if "stt" in MODEL_CACHE:
        return MODEL_CACHE["stt"]

    # Téléchargement (cache local automatique)
    model_dir = hf_hub_download(MODEL_REPO, "", subfolder=SUBDIR, cache_dir="./cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir).to(device)
    processor = AutoProcessor.from_pretrained(model_dir)

    stt_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor, device=0 if device=="cuda" else -1)

    MODEL_CACHE["stt"] = stt_pipeline
    return stt_pipeline

def transcribe_audio(audio_path: str):
    model = load_stt_model()
    return model(audio_path)["text"]
