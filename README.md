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

onnx_model = clip_onnx(model, source = "openai")
```
#### Open_CLIP Example
```python
import open_clip
from marqo_clip_onnx import clip_onnx

model, _, preprocess = open_clip.create_model_from_pretrained(model_name="ViT-L-14", pretrained="laion400m_e32",
                                                              device="cpu")

onnx_model = clip_onnx(model, source = "open_clip", device = "cpu")
```
### Step 2: ONNX Conversion

By default, we will export each onnx model into a folder, respectively. 
This is due to the limitation on the size of the onnx model, which can not exceed 2GB (details can be found [here](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md)).
In this circumstance, multiple files will be generated during the conversion. For models that are smaller than 2GB, only one single
`.onnx` file will be generated in the folder. 

```python



onnx_model.convert2onnx(visual_input=image, textual_input=text, verbose=True)

```