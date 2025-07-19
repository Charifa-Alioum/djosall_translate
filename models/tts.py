from huggingface_hub import hf_hub_download
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
import os, uuid

MODEL_CACHE = {}
MODEL_REPO = "Charifa/djos-all-models"
SUBDIR = "tts"

OUTPUT_DIR = "audio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tts_model():
    if "tts" in MODEL_CACHE:
        return MODEL_CACHE["tts"]

    model_dir = hf_hub_download(MODEL_REPO, "", subfolder=SUBDIR, cache_dir="./cache")
    model = VitsModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    MODEL_CACHE["tts"] = (model, tokenizer)
    return MODEL_CACHE["tts"]

def generate_tts_audio(text: str) -> str:
    model, tokenizer = load_tts_model()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.cpu().numpy().squeeze()
    audio = audio / np.max(np.abs(audio))

    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(OUTPUT_DIR, filename)
    write_wav(path, rate=model.config.sampling_rate, data=audio.astype(np.float32))
    return path
