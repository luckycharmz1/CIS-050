
import os

# Save in the current notebook's folder
output_folder = "./"  # or use "/content/"

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Setup
!pip install -q diffusers --upgrade
%%capture
!pip uninstall -y bitsandbytes
!pip install diffusers transformers accelerate --upgrade
!pip install -q invisible_watermark safetensors

from PIL import Image
from pprint import pprint

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Stable Diffusion - Generate Images with one object
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

prompt = "A portrait of kid in USA"
image = pipe(prompt).images[0]
image.show()

filename = output_folder + "/test_single.png"
image.save(filename)

# CLIP - evaluate image with one object
!pip install -q Pillow
import io
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
%%capture
cl_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
cl_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img_url = "./test_single.png"
raw_image = Image.open(img_url, mode='r')

# 1 or 2 word labels
captions = ["boy", "girl", "Man", "City"]
inputs = cl_processor(
        text=captions, images=raw_image, return_tensors="pt", padding=True
)
outputs = cl_model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print()
for i, caption in enumerate(captions):
   print('%40s - %.4f' % (caption, probs[0, i]))
print()

# Descriptive Captions
captions = ["A portarit of boy in a city",
            "happy girl in a city",
            "A walking old man",
            "Ape in the city Zoo"]

inputs = cl_processor(
        text=captions, images=raw_image, return_tensors="pt", padding=True
)
outputs = cl_model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print()
for i, caption in enumerate(captions):
   print('%40s - %.4f' % (caption, probs[0, i]))
print()

# Stable Diffusion - Object Counts
prompt = "3 cats with 2 chairs in the background with table and ball"
images = pipe(prompt=prompt).images[0]
filename = output_folder + "/test_counts.png"
images.save(filename)

# CLIP - evaluate object counts
img_url = './test_counts.png'
raw_image = Image.open(img_url, mode='r')

captions = ["Three cats",
            "Two cats",
            "Five cats",
            "Three balls"]

inputs = cl_processor(
        text=captions, images=raw_image, return_tensors="pt", padding=True
)
outputs = cl_model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print()
for i, caption in enumerate(captions):
   print('%40s - %.4f' % (caption, probs[0, i]))
print()
