import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os

class LLM:
    def _init_(self, model_path="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        # Load Hugging Face token from environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN environment variable not set. Model downloads for gated models might fail.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency on Ampere GPUs like A40 [4]
            device_map="auto", # Automatically distribute model across available GPUs [4]
            token=hf_token
        )
        self.device = device
        self.model.eval() # Set model to evaluation mode
        print(f"Llama LLM model loaded on {device}.") #[4]

        # Load LLM configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.max_new_tokens = config['llm']['max_new_tokens']
        self.temperature = config['llm']['temperature']
        self.top_p = config['llm']['top_p']
        
        # System message for conversational AI [5]
        self.system_message = {
            "role": "system",
            "content": "You are a helpful and empathetic AI assistant designed for phone calls. Respond naturally, concisely, and adapt your tone to the user's emotions. Maintain context across the conversation."
        }

    def generate_response(self, conversation_history, user_emotion=None):
        """
        Generates a text response based on conversation history and detected user emotion.
        conversation_history: List of dictionaries with "role" and "content" (text).
        user_emotion: String indicating detected emotion (e.g., "positive", "negative", "neutral").
        """
        # Prepare messages for LLM, including system message and conversation history [5]
        messages = [self.system_message] + conversation_history
        
        # Inject emotion into the prompt if available
        if user_emotion and user_emotion!= "neutral":
            # Find the last user message and append emotion context
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = f"{messages[i]['content']} (User's emotion: {user_emotion})"
                    break

        # Ensure conversation history doesn't exceed context window
        # For Llama 3.1, context length is typically 8K or more.
        # Manage conversation length to avoid errors and control costs. [5]
        # A simple truncation strategy: keep only the last N messages.
        max_history_length = 5 # Example: keep last 5 turns + system message
        if len(messages) > max_history_length + 1: # +1 for system message
            messages = [self.system_message] + messages[-(max_history_length):]

        # Apply chat template to format messages for the LLM
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id # Important for generation
            )
        
        # Decode only the newly generated tokens
        response = self.tokenizer.decode(output_ids[0, input_ids.shape:], skip_special_tokens=True)
        return response.strip()
