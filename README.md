# Marqo CLIP ONNX Converter

This is a simple library to convert pytorch clip model to their onnx versions.


# Examples

This library supports two major clip implementations, [openai clip]() and [open_clip]().


### Step 1: Model Loading
#### Openai CLIP Example

```python
import clip
from marqo_clip_onnx import clip_onnx


model, preprocess = clip.load(model_name="ViT-L-14", device="cpu", jit=False)
tokenizer = clip.tokenizer

onnx_model = clip_onnx(model, source = "openai")
```
#### Open_CLIP Example
```python
import open_clip
from marqo_clip_onnx import clip_onnx


model, _, preprocess = open_clip.create_model_from_pretrained(model_name="ViT-L-14", pretrained="laion400m_e32",
                                                              device="cpu")
tokenizer = open_clip.get_tokenizer("ViT-L-14")

onnx_model = clip_onnx(model, source = "open_clip", device = "cpu")
```
### Step 2: ONNX Conversion

By default, we will export each onnx model into a folder, respectively. 
This is due to the limitation on the size of the onnx model, which can not exceed 2GB (details can be found [here](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md)).
In this circumstance, multiple files will be generated during the conversion. For models that are smaller than 2GB, only one single
`.onnx` file will be generated in the folder. 

```python
import numpy as np
from PIL import Image

dummy = np.random.rand(1000, 800, 3) * 255
dummy = dummy.astype("uint8")
dummy_image = preprocess(Image.fromarray(dummy).convert("RGB")).unsqueeze(0).cpu()

dummy_text = tokenizer(["a diagram", "a dog", "a cat"]).cpu()

onnx_model.convert_onnx32(visual_input=dummy_image, textual_input=dummy_text, verbose=True)
# If you also need onnx16 version
onnx_model.convert_onnx16()
```
### Step 3: Check Difference

In this step, we compute the *normalized sum difference* between the outputs of the generated onnx models and the original torch model. This is to verify the
correctness of our onnx models.

```python
image = Image.open("coco.jpg")
text = "a horse carrying a large load of hay and two people sitting on it"

onnx_model.check_difference(exampl_image = image, example_text = text)
```