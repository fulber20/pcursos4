# gradio_interface.py
import gradio as gr
from audio_processing import preprocess_audio
from chatbot import create_chatbot
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Crear interfaz
def all_together(audio):
    conversation = create_chatbot()
    processor, model = load_seamless_model()
    arr_audio = preprocess_audio(audio, processor, model)
    query_input = speech_to_text(arr_audio)
    llm_response, last_response = generate_llm_response(query_input)
    spanish_output, english_output = text_to_speech(last_response)
    english_text = text_to_text(last_response)
    return llm_response, spanish_output, english_output, english_text

# Configuración de Gradio
iface = gr.Interface(
    fn=all_together,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Chat"),
        gr.Audio(label="Audio en Español", autoplay=False),
        gr.Audio(label="Audio en Ingles", autoplay=False),
        gr.Markdown(label="Traduccion"),
    ],
    title="Tutor con AI para practicar tu ingles",
    description="Graba tu voz usando el micrófono"
)

iface.launch(debug=False)

