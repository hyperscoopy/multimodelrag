from langchain_nvidia_ai_endpoints import ChatNVIDIA
from .config import NVIDIA_MODEL_NAME

def initialize_llm():

    llm = ChatNVIDIA(
        model=NVIDIA_MODEL_NAME,
        temperature=0.1
    )

    return llm
