# Building Blocks for Multi-Modal Apps Powered by Gradio and Groq

[Video Demo](https://github.com/user-attachments/assets/0ab0f71a-4b0a-4d58-ae79-02573aa8a21d)

This repository includes an application showing how to build fast multi-modal apps on Gradio powered by Groq. Specifically, it uses Whisper and Llama-3.2-vision to enable voice to text to LLM response, image to text, and traditional chat.

### Quickstart

To run the Gradio app, follow these instructions:

~~~
python3 -m venv venv
~~~

~~~
source venv/bin/activate
~~~

~~~
pip3 install -r requirements.txt
~~~

~~~
export GROQ_API_KEY=gsk...
~~~

~~~
python3 app.py
~~~

And your app will be hosted at http://127.0.0.1:7860!
