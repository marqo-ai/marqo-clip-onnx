import onnx
from clip_onnx import clip_onnx
import clip
import open_clip
import numpy as np
from PIL import Image
from onnxmltools.utils import float16_converter


def main():
    SOURCE = "openai" #or "open-clip"
    MODEL_NAME= "ViT-L/14"

    f32_VISUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
    f32_TEXTUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"

    f16_VISUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
    f16_TEXTUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"


    if SOURCE == "openai":
        model, preprocess = clip.load(name = MODEL_NAME, device = "cpu", jit=False)
        tokenize = clip.tokenize

    dummy = np.random.rand(1000,800,3) * 255
    dummy = dummy.astype("uint8")

    image = preprocess(Image.fromarray(dummy).convert("RGB")).unsqueeze(0).cpu()
    image_onnx = image.detach().cpu().numpy().astype(np.float32)

    text = tokenize(["a diagram", "a dog", "a cat"]).cpu()
    text_onnx = text.detach().cpu().numpy().astype(np.int32)

    onnx_model = clip_onnx(model, visual_path=f32_VISUAL_PATH, textual_path=f32_TEXTUAL_PATH)
    onnx_model.convert2onnx(visual_input=image, textual_input=text, verbose=True)

    f16_visual_model = float16_converter.convert_float_to_float16_model_path(
        f32_VISUAL_PATH
    )
    f16_textual_model= float16_converter.convert_float_to_float16_model_path(
        f32_TEXTUAL_PATH
    )

    onnx.save_model(f16_visual_model, f16_VISUAL_PATH)
    onnx.save_model(f16_textual_model, f16_TEXTUAL_PATH)

if __name__ == "__main__":
    main()