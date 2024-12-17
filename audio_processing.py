# audio_processing.py
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Definir el dispositivo (GPU o CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Definir la tasa de muestreo y la longitud máxima del audio
AUDIO_SAMPLE_RATE = 16000  # Frecuencia de muestreo deseada
MAX_INPUT_AUDIO_LENGTH = 30  # Longitud máxima en segundos

# Cargar el modelo Seamless
def load_seamless_model():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
    return processor, model

# Preprocesar el audio para Seamless
def preprocess_audio(input_audio: str, processor, model):
    print(f"Cargando audio desde: {input_audio}")  # Verificar la ruta
    arr, org_sr = torchaudio.load(input_audio)  # Cargar el archivo de audio
    print(f"Audio cargado con frecuencia de muestreo original: {org_sr}")
    
    # Re-muestrear el audio a la frecuencia deseada
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    print(f"Audio re-muestreado con frecuencia de {AUDIO_SAMPLE_RATE} Hz")
    
    # Recortar el audio si excede la longitud máxima
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
    
    # Preprocesar el audio con el modelo
    audio_inputs = processor(audios=new_arr, return_tensors="pt").to(device)
    return audio_inputs
