import torch
import numpy as np
import torchaudio

class VAD:
    def _init_(self, sample_rate=16000, vad_threshold=0.5, min_speech_duration=0.1, min_silence_duration=0.5, device="cpu"):
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=False,
                                                onnx=False)
        self.model = self.model.to(device)
        self.vad_threshold = vad_threshold
        self.sample_rate = sample_rate
        self.device = device
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils
        self.vad_iterator = self.VADIterator(self.model, sampling_rate=self.sample_rate)
        print(f"VAD model loaded on {device}.")

    def reset_states(self):
        """Resets the internal states of the VAD iterator."""
        self.vad_iterator.reset_states()

    def detect(self, audio_chunk_np):
        """
        Detects speech activity in a small audio chunk using Silero VADIterator.
        Returns a dictionary with 'start' and 'end' if speech is detected, otherwise None.
        """
        audio_tensor = torch.from_numpy(audio_chunk_np).float().to(self.device)
        
        # Silero VADIterator processes chunks and returns speech timestamps when an utterance is complete
        # or when silence is detected after speech.
        speech_dict = self.vad_iterator(audio_tensor, return_seconds=True)
        
        return speech_dict
