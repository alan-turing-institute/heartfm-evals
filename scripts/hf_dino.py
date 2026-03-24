"""
HuggingFace DINOv3 Feature Extraction Demo

Extracts features from an image using the DINOv3 ConvNext model via the
HuggingFace transformers pipeline.
"""
from huggingface_hub import login
from transformers import pipeline
from transformers.image_utils import load_image

# Set your HuggingFace token here or via the HF_TOKEN environment variable
login(token="")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction",
)
features = feature_extractor(image)

print(f"Feature length: {len(features[0][0])}")
