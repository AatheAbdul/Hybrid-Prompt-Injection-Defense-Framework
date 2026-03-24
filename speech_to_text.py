import os
import torch
from faster_whisper import WhisperModel
import numpy as np

# ----------------------------------------------------
# Autodetect Hardware
# ----------------------------------------------------
# Global variable for lazy loading
_stt_model_instance = None

def get_stt_model():
    global _stt_model_instance
    if _stt_model_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Audio] Loading Whisper STT Engine (Device: {device.upper()})...")
        _stt_model_instance = WhisperModel(
            "base",
            device=device,
            compute_type="int8" if device == "cpu" else "float16"
        )
    return _stt_model_instance


# ----------------------------------------------------
# Speech to Text Function
# ----------------------------------------------------

def transcribe_audio(audio_path):
    """
    Convert speech audio to text using faster-whisper.
    """
    if not os.path.exists(audio_path):
        return "", {"error": "Audio file not found"}

    try:
        model = get_stt_model()
        # Transcribe audio file
        segments, info = model.transcribe(audio_path, beam_size=5)

        transcript = []
        confidences = []

        for segment in segments:
            transcript.append(segment.text)
            
            # Convert logprob to 0.0-1.0 confidence score
            # Exp(logprob) gives a rough probability estimation
            conf_score = min(1.0, np.exp(segment.avg_logprob))
            confidences.append(float(conf_score))

        full_text = " ".join(transcript).strip()

        metadata = {
            "num_segments": len(confidences),
            "language": info.language,
            "language_probability": info.language_probability,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "status": "success"
        }

        return full_text, metadata

    except Exception as e:
        return "", {"error": str(e), "status": "failed"}


# ----------------------------------------------------
# Standalone test
# ----------------------------------------------------

if __name__ == "__main__":
    audio_path = input("Enter audio file path: ").strip().strip('"')

    text, meta = transcribe_audio(audio_path)

    if "error" in meta:
        print(f"[Error] {meta['error']}")
    else:
        print("\n[Audio] Transcribed Text:")
        print(text if text else "[Silence detected]")
        print("\n[Audio] STT Metadata:")
        print(meta)