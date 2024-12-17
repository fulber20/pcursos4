# main.py
import gradio as gr
from audio_processing import preprocess_audio, load_seamless_model
from chatbot import create_chatbot
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Función principal que integra todo
def all_together(audio):
    conversation = create_chatbot()  # Crear el chatbot
    processor, model = load_seamless_model()  # Cargar el modelo
    arr_audio = preprocess_audio(audio, processor, model)  # Preprocesar el audio

    # Aquí necesitarías una función para convertir audio en texto (por ejemplo, speech-to-text)
    query_input = speech_to_text(arr_audio)  # Convierte audio a texto (función no proporcionada)
    
    # Generar respuesta del chatbot
    llm_response, last_response = generate_llm_response(query_input)  # Necesitas definir esta función
    
    # Traducir el texto (esto también debe estar implementado)
    spanish_output, english_output = text_to_speech(last_response)  # Convierte texto a audio en español e inglés
    english_text = text_to_text(last_response)  # Convierte la respuesta a texto en inglés
    
    return llm_response, spanish_output, english_output, english_text

# Función de interfaz con Gradio
iface = gr.Interface(
    fn=all_together,
    inputs=gr.Audio(type="filepath"),  # Entrada de audio (archivo)
    outputs=[
        gr.Textbox(label="Chat"),
        gr.Audio(label="Audio en Español", autoplay=False),
        gr.Audio(label="Audio en Ingles", autoplay=False),
        gr.Markdown(label="Traducción"),
    ],
    title="Eleva tu inglés con un tutor virtual avanzado",
    description="Graba tu voz usando el micrófono"
)

# Lanzar la interfaz de Gradio
iface.launch(debug=True)
