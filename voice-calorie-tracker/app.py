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

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
if not api_key or not cartesia_api_key:
    raise ValueError("Please set both GROQ_API_KEY and CARTESIA_API_KEY environment variables.")
client = groq.Client(api_key=api_key)

async def text_to_speech(text: str) -> tuple[int, np.ndarray]:
    """Convert text to speech using Cartesia AI API"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cartesia.ai/tts/bytes",
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
            
            audio_bytes = await response.read()
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            return 24000, audio_array

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
            "content": "In conversation with the user, ask questions to estimate and provide (1) total calories, (2) protein, carbs, and fat in grams, (3) fiber and sugar content. Only ask *one question at a time*. Be conversational and natural.",
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
def response(state: AppState, audio: tuple):
    if not audio:
        return AppState()

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

        # Convert assistant's response to speech
        try:
            sample_rate, audio_array = asyncio.run(text_to_speech(assistant_message))
            state.last_audio_response = (sample_rate, audio_array)
        except Exception as e:
            print(f"Error in TTS: {e}")
            state.last_audio_response = None

        # Append assistant's message
        state.conversation.append({"role": "assistant", "content": assistant_message})
        
        print(state.conversation)
        os.remove(file_name)

    return state, state.conversation, state.last_audio_response

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
        output_audio = gr.Audio(
            label="AI Response",
            type="numpy",
            autoplay=True,
            visible=True,
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
    demo.launch()