from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, 
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory  # Corregí el nombre de ConversationBufferWindowMemory
import torch
import logging
import scipy
import gradio as gr
from getpass import getpass
import os
# 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
# Solicitar la API Key de OpenAI
OPENAI_API_KEY = getpass('Enter the secret value: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY  # Corregí el nombre de 'envirion' a 'environ'
llm = ChatOpenAI(model='gpt-4')
promot_system ='''acuta como un profesor de ingles, tu trabajo es enseñar y dejar ejercicos practicos.

responde siempre en inges y recuerda que le enseñes a alguien que habla español, se muy breve y puntual en tu respuesta
'''
prompt =ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
             prompt_system
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
nversationBufferWindowMemory(memory_key="chat_history", return_massages=True k=3)
n = LLMChain(llm=llm, prompt=prompt, memory=memory)