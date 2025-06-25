import torch
import numpy as np
import yaml
import queue
import threading
import time
import base64
import json
import os
from pydub import AudioSegment
from pydub.playback import play # For local testing/debugging audio playback
import soundfile as sf
from dotenv import load_dotenv

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import modules
from src.vad_module import VAD
from src.asr_module import ASR
from src.emotion_module import EmotionDetector
from src.llm_module import LLM
from src.tts_module import TTS
from src.voice_cloning_utils import VoiceCloner

# Load environment variables (e.g., HF_TOKEN)
load_dotenv()

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('config/model_paths.yaml', 'r') as f:
    model_paths = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize Flask app and SocketIO
app = Flask(_name_, template_folder='../templates')
app.config = 'your_secret_key_here' # Replace with a strong secret key
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # Use eventlet for async WSGI server

# Global instances of models (loaded once)
vad_model = VAD(
    sample_rate=config['audio']['sample_rate'],
    vad_threshold=config['audio']['vad_threshold'],
    min_speech_duration=config['audio']['vad_min_speech_duration'],
    min_silence_duration=config['audio']['vad_min_silence_duration'],
    device=DEVICE
)
asr_model = ASR(model_path=model_paths['whisper_path'], device=DEVICE)
emotion_detector = EmotionDetector(device=DEVICE)
llm_model = LLM(model_path=model_paths['llama_path'], device=DEVICE)
tts_model = TTS(csm_path=model_paths['csm_path'], mimi_path=model_paths['mimi_path'], device=DEVICE)
voice_cloner = VoiceCloner(tts_model, asr_model, sample_rate=config['audio']['sample_rate'], device=DEVICE)

# Store conversation history and VAD state per session ID
session_data = {} # {sid: {'history':, 'vad_buffer':, 'vad_iterator': VADIterator}}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    session_data[sid] = {
        'history':,
        'vad_buffer':,
        'vad_iterator': vad_model.VADIterator(vad_model.model, sampling_rate=vad_model.sample_rate),
        'last_speech_time': time.time(),
        'is_speaking': False,
        'full_utterance_audio':
    }
    emit('response', {'text': 'Hello! How can I help you today?'})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in session_data:
        del session_data[sid]

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    if sid not in session_data:
        print(f"No session data for {sid}, reconnecting or error.")
        return

    # Decode base64 audio chunk
    audio_bytes = base64.b64decode(data['audio'])
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0 # Normalize to float32

    current_session = session_data[sid]
    vad_iterator = current_session['vad_iterator']
    full_utterance_audio = current_session['full_utterance_audio']
    is_speaking = current_session['is_speaking']
    last_speech_time = current_session['last_speech_time']

    # VAD: Detect speech activity [11, 12]
    audio_tensor = torch.from_numpy(audio_np).float().to(DEVICE)
    speech_dict = vad_iterator(audio_tensor, return_seconds=True)
    current_time = time.time()

    if speech_dict: # Silero VADIterator returns dict if speech is detected
        if not is_speaking:
            print(f"[{sid}] VAD: Speech detected (start of utterance)")
            is_speaking = True
        full_utterance_audio.append(audio_tensor)
        last_speech_time = current_time
    elif is_speaking and (current_time - last_speech_time > config['audio']['vad_min_silence_duration']):
        print(f"[{sid}] VAD: End of utterance (processing)")
        is_speaking = False
        if full_utterance_audio:
            # Process the full utterance in a separate thread to avoid blocking the SocketIO event loop
            threading.Thread(target=process_full_utterance, args=(sid, torch.cat(full_utterance_audio))).start()
            current_session['full_utterance_audio'] = # Reset for next utterance
            vad_iterator.reset_states() # Reset VAD state for next utterance
    elif not is_speaking and full_utterance_audio and (current_time - last_speech_time > config['audio']['vad_min_silence_duration'] * 2):
        # Fallback: if VAD missed end, process anyway after a longer delay
        print(f"[{sid}] VAD: Force processing due to extended silence")
        threading.Thread(target=process_full_utterance, args=(sid, torch.cat(full_utterance_audio))).start()
        current_session['full_utterance_audio'] =
        vad_iterator.reset_states()

    current_session['is_speaking'] = is_speaking
    current_session['last_speech_time'] = last_speech_time
    current_session['full_utterance_audio'] = full_utterance_audio


def process_full_utterance(sid, audio_tensor):
    """Handles a complete user utterance: ASR, Emotion, LLM, TTS."""
    print(f"[{sid}] Processing full utterance...")
    audio_np = audio_tensor.cpu().numpy()

    # 1. ASR: Transcribe audio to text [13]
    user_text = asr_model.transcribe(audio_np)
    print(f"[{sid}] User: {user_text}")
    socketio.emit('transcript', {'text': user_text}, room=sid)

    if not user_text.strip():
        print(f"[{sid}] No meaningful speech detected, skipping LLM/TTS.")
        return

    # 2. Emotion Detection [1, 14]
    user_emotion_result = emotion_detector.get_simplified_emotion(user_text)
    user_emotion = user_emotion_result # e.g., "positive", "negative", "neutral"
    print(f"[{sid}] Detected emotion: {user_emotion}")

    # Update conversation history for LLM [5, 15]
    current_session = session_data[sid]
    current_session['history'].append({"role": "user", "content": user_text})

    # 3. LLM: Generate response [16, 5]
    llm_response_text = llm_model.generate_response(current_session['history'], user_emotion)
    print(f"[{sid}] Agent (text): {llm_response_text}")
    socketio.emit('response', {'text': llm_response_text}, room=sid)

    # Update conversation history with agent's response
    current_session['history'].append({"role": "assistant", "content": llm_response_text})

    # 4. TTS: Convert LLM response to speech with emotion/style [8, 6, 17]
    # Construct CSM-compatible conversation history for context [6]
    # For CSM, roles are typically "0" for user and "1" for assistant.
    csm_conversation_context =
    # Use last few turns as context for CSM to adapt style [8]
    for turn in current_session['history'][-4:]: # Use last 4 turns as context
        if turn["role"] == "user":
            csm_conversation_context.append({"role": "0", "content": [{"type": "text", "text": turn["content"]}]})
            # For voice cloning, you would pass the actual audio of the user's last utterance here
            # e.g., {"type": "audio", "path": audio_np} if you want to clone the user's voice for the agent's response.
            # For this example, we'll assume the agent speaks in its own voice.
        elif turn["role"] == "assistant":
            csm_conversation_context.append({"role": "1", "content": [{"type": "text", "text": turn["content"]}]})

    # Add the current response text as the final prompt for the assistant (speaker 1)
    csm_conversation_context.append({"role": "1", "content": [{"type": "text", "text": llm_response_text}]})

    # Generate speech. If voice cloning is desired, pass reference_audio_np to voice_cloner.
    # For now, we'll use the agent's default voice (speaker_id=1).
    # If you want to clone the user's voice for the agent's response:
    # agent_audio = voice_cloner.clone_and_synthesize(audio_np, llm_response_text)
    # Otherwise, use the default TTS model:
    agent_audio = tts_model.generate_speech(csm_conversation_context, speaker_id=config['tts']['speaker_id'])
    
    # Convert to bytes for sending over WebSocket
    agent_audio_bytes = (agent_audio * 32767).astype(np.int16).tobytes()
    agent_audio_b64 = base64.b64encode(agent_audio_bytes).decode('utf-8')

    # Send synthesized audio back to client
    socketio.emit('audio_response', {'audio': agent_audio_b64}, room=sid)
    print(f"[{sid}] Agent audio response sent.")


if _name_ == '_main_':
    # For local development, you might need to install eventlet: pip install eventlet
    # For production, use a proper WSGI server like Gunicorn with eventlet worker class:
    # gunicorn --worker-class eventlet -w 1 main:app --bind 0.0.0.0:5000
    print(f"Starting Flask-SocketIO server on port {config['web_server']['port']}...")
    socketio.run(app, host='0.0.0.0', port=config['web_server']['port'], debug=True
