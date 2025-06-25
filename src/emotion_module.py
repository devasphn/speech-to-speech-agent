import torch
from transformers import pipeline
import numpy as np

class EmotionDetector:
    def _init_(self, device="cpu"):
        self.device = device
        # Using a sentiment analysis model as a proxy for emotion detection
        # This model classifies text into POSITIVE, NEGATIVE, NEUTRAL.
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english", #S_S9, S_S10
            device=0 if device == "cuda" else -1 # 0 for GPU, -1 for CPU
        )
        print(f"EmotionDetector (Sentiment Analysis) initialized on {device}.") #[1, 14]

    def detect_emotion(self, text_input):
        """
        Detects emotion/sentiment from text.
        Returns a dictionary with 'label' (e.g., POSITIVE, NEGATIVE, NEUTRAL) and 'score'.
        """
        if not text_input.strip():
            return {"label": "NEUTRAL", "score": 1.0}

        # The pipeline returns a list of dictionaries, e.g.,
        result = self.sentiment_pipeline(text_input) #S_S10
        return result

    def get_simplified_emotion(self, text_input):
        """
        Returns a simplified emotion label (e.g., "positive", "negative", "neutral").
        """
        detection_result = self.detect_emotion(text_input)
        label = detection_result['label']
        
        if label == "POSITIVE":
            return "positive"
        elif label == "NEGATIVE":
            return "negative"
        else: # Includes "NEUTRAL"
            return "neutral"
