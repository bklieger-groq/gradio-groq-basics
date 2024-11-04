import io
import os
import time
import traceback
from dataclasses import dataclass, field
import groq
import aiohttp
import asyncio
import numpy as np
import soundfile as sf
import gradio as gr
import librosa
import spaces
import xxhash
from datasets import Audio
import json
import base64
from gradio_webrtc import WebRTC

# Instruction (This is what customizes the app to be a calorie tracker)
SYS_PROMPT = "In conversation with the user, ask questions to estimate and provide (1) total calories, (2) protein, carbs, and fat in grams, (3) fiber and sugar content. Only ask *one question at a time*. Be conversational and natural."

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
if not api_key or not cartesia_api_key:
    raise ValueError("Please set both GROQ_API_KEY and CARTESIA_API_KEY environment variables.")
client = groq.Client(api_key=api_key)

def parse_sse_event(event_str):
    event = {}
    for line in event_str.splitlines():
        if line.startswith(':'):
            continue  # Ignore comments
        if ':' in line:
            key, value = line.split(':', 1)
            value = value.lstrip()
            if key in event:
                event[key] += '\n' + value
            else:
                event[key] = value
    return event

async def text_to_speech_stream(text: str):
    """Stream audio chunks using Cartesia AI's streaming TTS API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cartesia.ai/tts/sse",
            headers={
                "Cartesia-Version": "2024-06-30",
                "Content-Type": "application/json",
                "X-API-Key": cartesia_api_key,
            },
            json={
                "model_id": "sonic-english",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": "79a125e8-cd45-4c13-8a67-188112f4dd22",
                },
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 24000,
                },
            },
        ) as response:
            if response.status != 200:
                raise Exception(f"TTS API error: {await response.text()}")

            buffer = ''
            sample_rate = 24000
            async for chunk, _ in response.content.iter_chunks():
                chunk_text = chunk.decode('utf-8')
                buffer += chunk_text
                while '\n\n' in buffer:
                    event_str, buffer = buffer.split('\n\n', 1)
                    event = parse_sse_event(event_str)
                    if 'data' in event:
                        data_str = event['data']
                        try:
                            data_json = json.loads(data_str)
                            if data_json.get('type') == 'chunk':
                                # Decode base64 audio data
                                chunk_data_base64 = data_json.get('data')
                                chunk_audio_bytes = base64.b64decode(chunk_data_base64)
                                # Convert to NumPy array
                                audio_array = np.frombuffer(chunk_audio_bytes, dtype=np.float32)
                                # Yield the audio chunk
                                yield sample_rate, audio_array
                            if data_json.get('done'):
                                # Streaming is complete
                                return
                        except Exception as e:
                            print(f"Error parsing data: {e}")
                            continue

def process_whisper_response(completion):
    """Process Whisper transcription response and return text or null based on no_speech_prob"""
    if completion.segments and len(completion.segments) > 0:
        no_speech_prob = completion.segments[0].get('no_speech_prob', 0)
        print("No speech prob:", no_speech_prob)

        if no_speech_prob > 0.7:
            return None
            
        return completion.text.strip()
    
    return None

def transcribe_audio(client, file_name):
    if file_name is None:
        return None

    try:
        with open(file_name, "rb") as audio_file:
            response = client.audio.transcriptions.with_raw_response.create(
                model="whisper-large-v3-turbo",
                file=("audio.wav", audio_file),
                response_format="verbose_json",
            )
            completion = process_whisper_response(response.parse())
            print(completion)
            
        return completion
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error in transcription: {str(e)}"

def generate_chat_completion(client, history):
    messages = []
    messages.append(
        {
            "role": "system",
            "content": SYS_PROMPT,
        }
    )

    for message in history:
        messages.append(message)

    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
        )
        assistant_message = completion.choices[0].message.content
        return assistant_message
    except Exception as e:
        return f"Error in generating chat completion: {str(e)}"

@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    stopped: bool = False
    model_outs: any = None
    last_audio_response: tuple = None

def process_audio(audio: tuple, state: AppState):
    return audio, state

@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
async def response(state: AppState, audio: tuple):
    if not audio:
        yield state, state.conversation, None
        return

    file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav"
    sf.write(file_name, audio[1], audio[0], format="wav")

    # Transcribe the audio file
    transcription = transcribe_audio(client, file_name)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        # Append user's message
        state.conversation.append({"role": "user", "content": transcription})

        # Generate assistant response
        assistant_message = generate_chat_completion(client, state.conversation)

        # Append assistant's message
        state.conversation.append({"role": "assistant", "content": assistant_message})

        # Update the conversation
        yield state, state.conversation, None  # Update chatbot without audio

        # Stream TTS audio through WebRTC
        try:
            audio_generator = text_to_speech_stream(assistant_message)
            async for audio_chunk in audio_generator:
                # Yield the raw audio bytes to the output_audio component
                yield state, state.conversation, audio_chunk
        except Exception as e:
            print(f"Error in TTS: {e}")
            yield state, state.conversation, None

        print(state.conversation)
        os.remove(file_name)
    else:
        yield state, state.conversation, None

def start_recording_user(state: AppState):
    return None

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

# Modify the output_audio component to use WebRTC
with gr.Blocks(theme=theme, js=js) as demo:
    with gr.Row():
        input_audio = gr.Audio(
            label="Input Audio",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", type="messages")
    with gr.Row():
        output_audio = gr.WebRTC(
            label="AI Response",
            mode="recvonly",  # Only receiving audio from the server
            media_stream_constraints={"audio": True, "video": False},
        )

    state = gr.State(value=AppState())
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        response, [state, input_audio], [state, chatbot, output_audio]
    )
    restart = respond.then(start_recording_user, [state], [input_audio]).then(
        lambda state: state, state, state, js=js_reset
    )

    cancel = gr.Button("New Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False), None),
        None,
        [state, input_audio, output_audio],
        cancels=[respond, restart],
    )

if __name__ == "__main__":
    demo.queue().launch()