# from groq import Groq

# client = Groq()
# completion = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {
#             "role": "system",
#             "content": "hello\n"
#         },
#         {
#             "role": "user",
#             "content": "hello\n"
#         },
#         {
#             "role": "assistant",
#             "content": "It's nice to meet you. Is there something I can help you with or would you like to chat?"
#         }
#     ],
#     temperature=1,
#     max_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")

import gradio as gr
import groq
import io
import numpy as np
import soundfile as sf
api_key = "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"
def transcribe_audio(audio, api_key):
    if audio is None:
        return ""
    
    client = groq.Client(api_key=api_key)
    
    # Convert audio to the format expected by the model
    # The model supports mp3, mp4, mpeg, mpga, m4a, wav, and webm file types 
    audio_data = audio[1]  # Get the numpy array from the tuple
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, audio[0], format='wav')
    buffer.seek(0)

    bytes_audio = io.BytesIO()
    np.save(bytes_audio, audio_data)
    bytes_audio.seek(0)

    try:
        # Use Distil-Whisper English powered by Groq for transcription
        completion = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en",
            file=("audio.wav", buffer),
            response_format="text"
        )
        return completion
    except Exception as e:
        return f"Error in transcription: {str(e)}"
    
def generate_response(transcription, api_key):
    if not transcription:
        return "No transcription available. Please try speaking again."
    
    client = groq.Client(api_key=api_key)
    
    try:
        # Use Llama 3 70B powered by Groq for text generation
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in response generation: {str(e)}"
    
def process_audio(audio, api_key):
    if not api_key:
        return "Please enter your Groq API key.", "API key is required."
    transcription = transcribe_audio(audio, api_key)
    response = generate_response(transcription, api_key)
    return transcription, response

# Custom CSS for the Groq badge and color scheme (feel free to edit however you wish)
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button-primary {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
.gr-button-secondary {
    color: #f55036 !important;
    border-color: #f55036 !important;
}
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎙️ Groq x Gradio Voice-Powered AI Assistant")
    text_input = gr.Textbox()
    api_key_input = gr.Textbox(type="password", label="Enter your Groq API Key")
    
    with gr.Row():
        audio_input = gr.Audio(label="Speak!", type="numpy")
    
    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription")
        response_output = gr.Textbox(label="AI Assistant Response")
    
    submit_button = gr.Button("Process", variant="primary")
    
    # Add the Groq badge
    gr.HTML("""
    <div id="groq-badge">
        <div style="color: #f55036; font-weight: bold;">POWERED BY GROQ</div>
    </div>
    """)
    
    submit_button.click(
        generate_response,
        inputs=[text_input, api_key_input],
        outputs=[ response_output]
    )
    
    gr.Markdown("""
    ## How to use this app:
    1. Enter your Groq API Key in the provided field.
    2. Click on the microphone icon and speak your message (or forever hold your peace)! You can also provide a supported audio file. Supported audio files include mp3, mp4, mpeg, mpga, m4a, wav, and webm file types.
    3. Click the "Process" button to transcribe your speech and generate a response from our AI assistant.
    4. The transcription and AI assistant response will appear in the respective text boxes.
    
    """)

demo.launch()