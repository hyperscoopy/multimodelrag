import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .config import CLIP_MODEL_NAME


# Initialize CLIP Model
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.eval()


def embed_image(image_data):
    """
    Embed an image using CLIP.
    """

    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = clip_model.get_image_features(
            **inputs
        )

        # Handle Transformers output object
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        # Normalize embedding
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        return features.squeeze().numpy()


def embed_text(text):
    """
    Embed text using CLIP.
    """

    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():

        outputs = clip_model.get_text_features(
            **inputs
        )

        # Handle Transformers output object
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        # Normalize embedding
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        return features.squeeze().numpy()