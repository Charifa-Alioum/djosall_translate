import os
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Cache global des modèles
MODEL_CACHE = {}
MODEL_REPO = "Charifa/djos-all-models"

# Mapping source->target vers dossier
MODEL_MAP = {
    ("en", "ff"): "en_ff",   # ONNX
    ("ff", "en"): "ff_en",   # ONNX
    ("fr", "ff"): "fr_ff",   # PyTorch (Transformers)
    ("ff", "fr"): "ff_fr"    # PyTorch (Transformers)
}

def load_translation_model(src: str, tgt: str):
    """
    Charge dynamiquement un modèle de traduction (ONNX ou PyTorch) depuis Hugging Face avec cache.
    """
    key = (src, tgt)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if key not in MODEL_MAP:
        raise ValueError(f"Traduction non supportée: {src}->{tgt}")

    subdir = MODEL_MAP[key]

    # Cas 1 : Modèles ONNX (en_ff, ff_en)
    if subdir in ["en_ff", "ff_en"]:
        encoder_path = hf_hub_download(MODEL_REPO, "encoder_model.onnx", subfolder=subdir, cache_dir="./cache")
        decoder_path = hf_hub_download(MODEL_REPO, "decoder_model.onnx", subfolder=subdir, cache_dir="./cache")
        source_spm = hf_hub_download(MODEL_REPO, "source.spm", subfolder=subdir, cache_dir="./cache")
        target_spm = hf_hub_download(MODEL_REPO, "target.spm", subfolder=subdir, cache_dir="./cache")

        # Tokenizers
        sp_source = spm.SentencePieceProcessor(model_file=source_spm)
        sp_target = spm.SentencePieceProcessor(model_file=target_spm)

        # Sessions ONNX
        encoder_sess = ort.InferenceSession(encoder_path)
        decoder_sess = ort.InferenceSession(decoder_path)

        MODEL_CACHE[key] = {
            "type": "onnx",
            "encoder": encoder_sess,
            "decoder": decoder_sess,
            "sp_source": sp_source,
            "sp_target": sp_target
        }
        return MODEL_CACHE[key]

    # Cas 2 : Modèles Transformers (fr_ff, ff_fr)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_REPO}/{subdir}", cache_dir="./cache")
        model = AutoModelForSeq2SeqLM.from_pretrained(f"{MODEL_REPO}/{subdir}", cache_dir="./cache", torch_dtype=torch.float32)
        model.eval()

        MODEL_CACHE[key] = {
            "type": "transformers",
            "tokenizer": tokenizer,
            "model": model
        }
        return MODEL_CACHE[key]

def translate_text(text: str, src: str, tgt: str):
    """
    Traduit un texte en utilisant soit ONNX soit Transformers selon le modèle.
    """
    model_data = load_translation_model(src, tgt)

    if model_data["type"] == "onnx":
        # ONNX pipeline
        sp_source, sp_target = model_data["sp_source"], model_data["sp_target"]
        encoder_sess, decoder_sess = model_data["encoder"], model_data["decoder"]

        tokens = sp_source.encode(text, out_type=int)
        tokens = np.array(tokens, dtype=np.int64)[None, :]

        encoder_outputs = encoder_sess.run(None, {"input_ids": tokens})[0]
        output_tokens = decoder_sess.run(None, {"encoder_hidden_states": encoder_outputs})[0]

        return sp_target.decode(output_tokens[0].tolist())

    else:
        # Transformers pipeline (PyTorch)
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
