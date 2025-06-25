import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import numpy as np
import yaml
import os

class TTS:
    def _init_(self, csm_path="sesame/csm-1b", mimi_path="sesame/mimi", device="cuda"):
        self.device = device
        # Load Hugging Face token from environment variable
        hf_token = os.getenv("HF_TOKEN")

        # CSM-1B is available natively in Hugging Face Transformers [6, 7]
        self.processor = AutoProcessor.from_pretrained(csm_path, token=hf_token)
        self.model = CsmForConditionalGeneration.from_pretrained(csm_path, token=hf_token).to(device)
        self.model.eval() # Set model to evaluation mode
        print(f"Sesame CSM-1B TTS model loaded on {device}.") #[6, 7]

        # Load TTS configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.speed_factor = config['tts']['speed_factor']
        self.speaker_id = config['tts']['speaker_id'] # Default speaker for agent

    def generate_speech(self, conversation_context, speaker_id=None):
        """
        Generates speech from text using CSM-1B, with context.
        conversation_context: Formatted list of dictionaries for CSM's chat template. [6]
                              Example: [{"role": "0", "content": [{"type": "text", "text": "Hello."}]}]
        speaker_id: The ID of the speaker for the generated response (e.g., 0 or 1).
        """
        if speaker_id is None:
            speaker_id = self.speaker_id

        # Prepare inputs for CSM, applying chat template for context [6]
        # The conversation_context is already formatted for CSM's apply_chat_template
        inputs = self.processor.apply_chat_template(
            conversation_context,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        # Generate audio. CSM sounds best with context. [8, 6]
        # The generate method takes output_audio=True to directly produce audio. [6]
        with torch.no_grad():
            audio_tensor = self.model.generate(
                **inputs,
                output_audio=True,
                # Add generation parameters if needed, e.g., temperature, top_k [9]
                # temperature=0.7, top_k=50,
                # For streaming, the 'csm-streaming-tf' repo explores decoding tokens. [10]
                # This current implementation is for full audio generation.
            )
        
        # The output audio_tensor is a raw audio tensor.
        # It needs to be converted to numpy for external use/playback.
        return audio_tensor.cpu().numpy()
