import torch
import numpy as np
import soundfile as sf
import torchaudio
from src.asr_module import ASR # Assuming ASR module is available for transcription

class VoiceCloner:
    def _init_(self, tts_model, asr_model, sample_rate=16000, device="cuda"):
        self.tts_model = tts_model # An instance of your TTS class (CSM-1B)
        self.asr_model = asr_model # An instance of your ASR class (Whisper)
        self.sample_rate = sample_rate
        self.device = device
        print(f"VoiceCloner initialized on {device}.") #[19]

    def clone_and_synthesize(self, reference_audio_np, text_to_synthesize):
        """
        Clones a voice from a reference audio (numpy array) and synthesizes new text.
        reference_audio_np: numpy array of reference audio.
        text_to_synthesize: The text to be spoken in the cloned voice.
        """
        print(f"Attempting to clone voice and synthesize: '{text_to_synthesize}'")
        
        # Ensure reference audio is 16kHz and mono
        if reference_audio_np.ndim > 1:
            reference_audio_np = reference_audio_np.mean(axis=0) # Convert to mono
        
        # 1. Transcribe reference audio for context (as suggested by phildougherty/sesame_csm_openai) [9]
        # This helps CSM better match the voice characteristics.
        ref_transcript = self.asr_model.transcribe(reference_audio_np)
        print(f"Reference audio transcribed: {ref_transcript}")

        # 2. Prepare context for CSM-1B with reference audio and its transcript [6, 9]
        # The first turn provides the reference audio for the voice to be cloned (speaker 0).
        # The second turn provides the text to be synthesized by that cloned voice.
        
        # CSM's apply_chat_template can take audio arrays in its content. [6]
        # The 'path' in 'type: "audio", "path": audio["array"]' for CSM context
        # refers to the raw audio array, not a file path. [6]
        
        cloning_context = [
            {"role": "0", "content": [{"type": "audio", "path": reference_audio_np}, {"type": "text", "text": ref_transcript}]},
            {"role": "0", "content": [{"type": "text", "text": text_to_synthesize}]} # Synthesize this text in cloned voice
        ]
        
        # Generate speech using the TTS model with the cloning context
        # The speaker_id for generation should match the speaker in the cloning_context (e.g., 0)
        generated_audio = self.tts_model.generate_speech(cloning_context, speaker_id=0)
        
        print("Voice cloning and synthesis complete.")
        return generated_audio
