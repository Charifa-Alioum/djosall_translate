import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

# Cache global des modèles
MODEL_CACHE = {}
MODEL_REPO = "Charifa/djos-all-models"

# Mapping source->target vers dossier
MODEL_MAP = {
    ("en", "ff"): "en_ff",
    ("ff", "en"): "ff_en",
    ("fr", "ff"): "fr_ff",
    ("ff", "fr"): "ff_fr"
}

def load_translation_model(src: str, tgt: str):
    """Charge dynamiquement un modèle de traduction ONNX depuis Hugging Face avec cache."""
    key = (src, tgt)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if key not in MODEL_MAP:
        raise ValueError(f"Traduction non supportée: {src}->{tgt}")

    subdir = MODEL_MAP[key]

    # Télécharger les fichiers requis
    encoder_path = hf_hub_download(MODEL_REPO, "encoder_model.onnx", subfolder=subdir, cache_dir="./cache")
    decoder_path = hf_hub_download(MODEL_REPO, "decoder_model.onnx", subfolder=subdir, cache_dir="./cache")
    source_spm = hf_hub_download(MODEL_REPO, "source.spm", subfolder=subdir, cache_dir="./cache")
    target_spm = hf_hub_download(MODEL_REPO, "target.spm", subfolder=subdir, cache_dir="./cache")

    # Charger tokenizers
    sp_source = spm.SentencePieceProcessor(model_file=source_spm)
    sp_target = spm.SentencePieceProcessor(model_file=target_spm)

    # Créer sessions ONNX
    encoder_sess = ort.InferenceSession(encoder_path)
    decoder_sess = ort.InferenceSession(decoder_path)

    # Sauvegarde cache
    MODEL_CACHE[key] = {
        "encoder": encoder_sess,
        "decoder": decoder_sess,
        "sp_source": sp_source,
        "sp_target": sp_target
    }
    return MODEL_CACHE[key]

def translate_text(text: str, src: str, tgt: str):
    """Traduit un texte en utilisant ONNX (pipeline simplifié)."""
    model = load_translation_model(src, tgt)
    sp_source, sp_target = model["sp_source"], model["sp_target"]
    encoder_sess, decoder_sess = model["encoder"], model["decoder"]

    # Encoder
    tokens = sp_source.encode(text, out_type=int)
    tokens = np.array(tokens, dtype=np.int64)[None, :]

    # Encoder output (selon signature exacte ONNX, ici simplifiée)
    encoder_outputs = encoder_sess.run(None, {"input_ids": tokens})[0]

    # Décoder (simplifié: un seul passage)
    output_tokens = decoder_sess.run(None, {"encoder_hidden_states": encoder_outputs})[0]

    return sp_target.decode(output_tokens[0].tolist())
