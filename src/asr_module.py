import torch
from transformers import pipeline
import numpy as np

class ASR:
    def _init_(self, model_path="openai/whisper-large-v3", device="cuda"):
        # Load Whisper model using Hugging Face pipeline for ease of use
        self.pipe = pipeline("automatic-speech-recognition", model=model_path, device=device)
        print(f"Whisper ASR model loaded on {device}.") #[18]

    def transcribe(self, audio_np):
        """
        Transcribes audio (numpy array) to text.
        """
        # The Whisper pipeline expects audio as a numpy array at 16kHz
        # Ensure the input audio_np is already at 16kHz
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0) # Convert to mono if stereo
        
        # Whisper is robust to accents due to its large training dataset [2, 3]
        result = self.pipe(audio_np)
        return result["text"]
