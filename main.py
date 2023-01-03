import onnx
from marqo_clip_onnx_converter import clip_converter
import clip
import open_clip
import numpy as np
from PIL import Image
from onnxmltools.utils import float16_converter


def conversion(SOURCE, MODEL_NAME, PRETRAINED):

    if SOURCE=="openai":

        f32_VISUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
        f32_TEXTUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"

        f16_VISUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
        f16_TEXTUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"

        model, preprocess = clip.load(name = MODEL_NAME, device = "cpu", jit=False)
        tokenizer = clip.tokenize

    elif SOURCE == "open_clip":
        f32_VISUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-visual.onnx"
        f32_TEXTUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-textual.onnx"

        f16_VISUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-visual.onnx"
        f16_TEXTUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-textual.onnx"

        model, _ , preprocess = open_clip.create_model_and_transforms(model_name = MODEL_NAME, device="cpu", pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        print(f'image mean: {getattr(model.visual,"image_mean", None)}')
        print(f'image std: {getattr(model.visual, "image_std", None)}')


    dummy = np.random.rand(1000,800,3) * 255
    dummy = dummy.astype("uint8")

    image = preprocess(Image.fromarray(dummy).convert("RGB")).unsqueeze(0).cpu()

    text = tokenizer(["a diagram", "a dog", "a cat"]).cpu()

    onnx_model = clip_converter(model, visual_path=f32_VISUAL_PATH, textual_path=f32_TEXTUAL_PATH, source= SOURCE)
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
    SOURCE = "open_clip" #or "open_clip"
    MODEL_NAME= "ViT-B-32-quickgelu"
    PRETRAINED = 'openai'  #only for open_clip

    conversion(SOURCE, MODEL_NAME, PRETRAINED)