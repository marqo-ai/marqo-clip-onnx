import clip
from marqo_clip_onnx import clip_onnx


model, preprocess = clip.load(name="ViT-B/32", device="cpu", jit=False)
tokenizer = clip.tokenize

onnx_model = clip_onnx(model, source = "openai")

import numpy as np
from PIL import Image

dummy = np.random.rand(1000, 800, 3) * 255
dummy = dummy.astype("uint8")
dummy_image = preprocess(Image.fromarray(dummy).convert("RGB")).unsqueeze(0).cpu()

dummy_text = tokenizer(["a diagram", "a dog", "a cat"]).cpu()


onnx_model.convert_onnx32(visual_input = dummy_image, textual_input=dummy_text, verbose=True)

# If you also need onnx16 version
onnx_model.convert_onnx16()


image = Image.open("coco.jpg")
text = "a horse carrying a large load of hay and two people sitting on it"

processed_image = preprocess(image).unsqueeze(0)
processed_text = tokenizer([text])

onnx_image = processed_image.detach().cpu().numpy()
onnx_text = processed_text.detach().cpu().numpy()

onnx_model.check_difference(image = processed_image, text = processed_text, onnx_image = onnx_image, onnx_text = onnx_text)