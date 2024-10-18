import gradio as gr
import groq
import os
import io
import numpy as np
import soundfile as sf
from PIL import Image
import base64

# Get the GROQ_API_KEY from environment variables
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

# Initialize the Groq client
client = groq.Client(api_key=api_key)

# Function to transcribe audio
def transcribe_audio(audio):
    if audio is None:
        return None
    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Normalize audio
    if np.max(np.abs(y)) != 0:
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
    else:
        y = y.astype(np.float32)

    # Write audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='wav')
    buffer.seek(0)

    try:
        # Use Distil-Whisper English model for transcription
        completion = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en",
            file=("audio.wav", buffer),
            response_format="text"
        )
        transcription = completion
        return transcription
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# Function to handle the chat conversation
def chat(audio_input, history):
    if history is None:
        history = []

    # Process user input
    user_input = None

    # Check if audio input is provided
    if audio_input is not None:
        transcription = transcribe_audio(audio_input)
        if transcription is not None:
            user_input = transcription
            # Append the transcription to the chat history
            history.append((user_input, None))  # User message, no image
        else:
            user_input = "Error in audio transcription."
            history.append((user_input, None))
        # Reset audio input after processing
        audio_input = None
    else:
        # No input provided
        return gr.update(), history, history

    # Prepare messages for the AI model
    messages = []
    messages.append({"role":"system", "content":"In conversation with the user, ask questions to provide a calorie count estimate. Only ask *one question at a time*."})
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Send to AI model
    try:
        # Use Llama-3.2 model for conversation
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
        )
        assistant_message = completion.choices[0].message.content
        history.append((None, assistant_message))  # Assistant message
    except Exception as e:
        assistant_message = f"Error: {str(e)}"
        history.append((None, assistant_message))

    # Update the chat history
    return gr.update(value=None), history, history

# Custom CSS for the interface
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
#bottom-image {
    text-align: center;
    margin-top: 20px;
}
#chatbox .wrap .message {
    border-radius: 10px;
    padding: 8px 12px;
    margin: 5px 0;
}
#chatbox .wrap .message.user {
    background-color: #d1e7dd;
    color: #000;
}
#chatbox .wrap .message.assistant {
    background-color: #fff;
    color: #000;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🍽️ Meal Calorie Breakdown")

    # Initialize chat history
    state = gr.State([])

    # Chatbot component
    chatbot = gr.Chatbot(elem_id="chatbox")

    # Start the conversation
    def start_conversation():
        initial_message = "Hello! What did you eat for your meal?"
        return [(None, initial_message)]  # Assistant message only

    chatbot.value = start_conversation()
    state.value = chatbot.value

    # Audio input component
    audio_input = gr.Audio(type="numpy", label="Talk about your meal", elem_id="audio-input")

    # Function to update the chat display without labels
    def format_chat(history):
        formatted_history = []
        for user_msg, assistant_msg in history:
            if user_msg:
                formatted_history.append((user_msg, None))
            if assistant_msg:
                formatted_history.append((None, assistant_msg))
        return formatted_history

    # Event handler for audio input
    audio_input.change(
        chat,
        inputs=[audio_input, state],
        outputs=[audio_input, chatbot, state],
        queue=False,
    )

    gr.Markdown("""
    ## How to use this app:

    - The assistant will start by asking you what you ate for your meal.
    - Respond by recording your answer using the audio recorder.
    - The assistant may ask follow-up questions based on your responses.
    - Continue the conversation as desired.
    """)

demo.launch()
