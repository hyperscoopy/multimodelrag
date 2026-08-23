import os

from dotenv import load_dotenv

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

NVIDIA_MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
PDF_PATH = "multimodal_sample.pdf"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

TOP_K = 5