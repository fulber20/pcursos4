import openai
from config import OPENAI_API_KEY  # Asegúrate de importar la clave desde config.py
openai.api_key = OPENAI_API_KEY
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

def create_chatbot():
    llm = ChatOpenAI(model='gpt-4')
    prompt_system = '''Actua como un profesor de ingles, tu trabajo es enseñar y dejar ejercicios practicos.
                       responde siempre en ingles y recuerda que le enseñas a alguien que habla español, se muy breve y puntual en tu respuesta'''

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(prompt_system),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return conversation
