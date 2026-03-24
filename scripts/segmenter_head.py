"""
Segmenter Head Demo

Demonstrates DINOv3 segmentation using the dinov3_vit7b16_ms model
with a Mask2Former head on an example image.
"""
import sys
from functools import partial

import matplotlib.pyplot as plt
import requests
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision.transforms import v2

REPO_DIR = "../models/dinov3/"
sys.path.append(REPO_DIR)

from dinov3.eval.segmentation.inference import make_inference


def get_img() -> Image.Image:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def make_transform(resize_size: int | list[int] = 768, use_fp16: bool = False):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    dtype = torch.float16 if use_fp16 else torch.float32
    to_float = v2.ToDtype(dtype, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


# -- Load segmentor model --
torch.backends.mps.enabled = True

segmentor = torch.hub.load(
    REPO_DIR,
    "dinov3_vit7b16_ms",
    source="local",
    weights="../model_weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    backbone_weights="../model_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
segmentor = segmentor.to(device)


# -- Run inference --
img_size = 896
img = get_img()
transform = make_transform(img_size, use_fp16=False)

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        batch_img = transform(img)[None].to(device)
        pred_vit7b = segmentor(batch_img)
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)


# -- Visualize results --
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0, 0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
plt.savefig("segmenter_head_result.png", dpi=150)
plt.close()
print("Saved segmenter_head_result.png")
