# instalar_requisitos.py
import os

# Instalación de los requisitos necesarios
os.system('pip install fairseq2')
os.system('pip install git+https://github.com/huggingface/transformers.git sentencepiece')
os.system('pip install -U transformers')
os.system('pip install -q langchain openai gradio')
