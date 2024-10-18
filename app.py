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

def transcribe_audio(audio):
    if audio is None:
        return "No audio provided.", ""
    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Normalize audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

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
    except Exception as e:
        transcription = f"Error in transcription: {str(e)}"

    response = generate_response(transcription)
    return transcription, response

def generate_response(transcription):
    if not transcription or transcription.startswith("Error"):
        return "No valid transcription available. Please try speaking again."

    try:
        # Use Llama 3.1 70B model for text generation
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in response generation: {str(e)}"

def analyze_image(image):
    if image is None:
        return "No image uploaded.", None

    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image.astype('uint8'), 'RGB')

    # Convert PIL image to base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        # Use Llama 3.2 11B Vision model to analyze image
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )
        description = chat_completion.choices[0].message.content
    except Exception as e:
        description = f"Error in image analysis: {str(e)}"

    return description

def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    # Prepare the message history for the API
    messages = []
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    try:
        # Use Llama 3.1 70B model for generating assistant response
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
        )
        assistant_message = completion.choices[0].message.content
        chat_history.append((message, assistant_message))
    except Exception as e:
        assistant_message = f"Error: {str(e)}"
        chat_history.append((message, assistant_message))

    return "", chat_history, chat_history  # Return state as the third output

# Custom CSS for the Groq badge and color scheme
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button-primary {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎙️ Groq x Gradio Multi-Modal Llama-3.2 and Whisper")

    with gr.Tab("Audio"):
        gr.Markdown("## Speak to the AI")
        with gr.Row():
            audio_input = gr.Audio(type="numpy", label="Speak or Upload Audio")
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription")
            response_output = gr.Textbox(label="AI Assistant Response")
        process_button = gr.Button("Process", variant="primary")
        process_button.click(
            transcribe_audio,
            inputs=audio_input,
            outputs=[transcription_output, response_output]
        )

    with gr.Tab("Image"):
        gr.Markdown("## Upload an Image for Analysis")
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
        with gr.Row():
            image_description_output = gr.Textbox(label="Image Description")
        analyze_button = gr.Button("Analyze Image", variant="primary")
        analyze_button.click(
            analyze_image,
            inputs=image_input,
            outputs=[image_description_output]
        )

    with gr.Tab("Chat"):
        gr.Markdown("## Chat with the AI Assistant")
        chatbot = gr.Chatbot()
        state = gr.State([])  # Initialize the chat state
        with gr.Row():
            user_input = gr.Textbox(show_label=False, placeholder="Type your message here...", container=False)
            send_button = gr.Button("Send", variant="primary")
        send_button.click(
            respond,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state],
        )

    # Add the Groq badge
    gr.HTML("""
    <div id="groq-badge">
        <div style="color: #f55036; font-weight: bold;">POWERED BY GROQ</div>
    </div>
    """)

    gr.Markdown("""
    ## How to use this app:

    ### Audio Tab
    1. Click on the microphone icon and speak your message or upload an audio file.
    2. Click the "Process" button to transcribe your speech and generate a response from the AI assistant.
    3. The transcription and AI assistant response will appear in the respective text boxes.

    ### Image Tab
    1. Upload an image by clicking on the image upload area.
    2. Click the "Analyze Image" button to get a detailed description of the image.
    3. The uploaded image and its description will appear below.

    ### Chat Tab
    1. Type your message in the text box at the bottom.
    2. Click the "Send" button to interact with the AI assistant.
    3. The conversation will appear in the chat interface.
    """)

demo.launch()
