# audio_processing.py
import torchaudio
import scipy.io.wavfile
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Cargar el modelo Seamless
def load_seamless_model():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
    return processor, model

# Preprocesar el audio para Seamless
def preprocess_audio(input_audio: str, processor, model):
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
    audio_inputs = processor(audios=new_arr, return_tensors="pt").to(device)
    return audio_inputs
