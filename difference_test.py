import open_clip
import numpy as np
import torch
import clip
import onnx
import onnxruntime as ort
from PIL import Image




def compute_dif(a, b):
    return np.abs(np.array(a) / np.linalg.norm(a) - np.array(np.array(b) / np.linalg.norm(b))).sum()


def onnx_evaluation(SOURCE, MODEL_NAME, PRETRAINED):
    if SOURCE == "openai":

        f32_VISUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
        f32_TEXTUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"

        f16_VISUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx"
        f16_TEXTUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx"

        model, preprocess = clip.load(name=MODEL_NAME, device="cpu", jit=False)
        tokenizer = clip.tokenize

    elif SOURCE == "open_clip":
        f32_VISUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-visual.onnx"
        f32_TEXTUAL_PATH = f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-textual.onnx"

        f16_VISUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-visual.onnx"
        f16_TEXTUAL_PATH = f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-{PRETRAINED}-textual.onnx"

        model, _, preprocess = open_clip.create_model_and_transforms(model_name=MODEL_NAME, device="cpu",
                                                                     pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    f16_textual_session = ort.InferenceSession(f16_TEXTUAL_PATH, providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])
    f16_visual_session =ort.InferenceSession(f16_VISUAL_PATH, providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])

    f32_textual_session =ort.InferenceSession(f32_TEXTUAL_PATH, providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])
    f32_visual_session =ort.InferenceSession(f32_VISUAL_PATH, providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])

    image = Image.open("coco.jpg")
    text = "a horse carrying a large load of hay and two people sitting on it"

    image_processed = preprocess(image).unsqueeze(0)
    text_processed = tokenizer([text])

    image_onnx = image_processed.detach().cpu().numpy()
    text_onnx = text_processed.detach().cpu().numpy()

    with torch.no_grad():
        torch_image_output = model.encode_image(image_processed).detach().cpu().numpy()
        torch_text_output = model.encode_text(text_processed).detach().cpu().numpy()

    f16onnx_image_output = f16_visual_session.run(None, {"input": image_onnx.astype(np.float16)})[0]
    f16onnx_text_output = f16_textual_session.run(None, {"input": text_onnx})[0]

    f32onnx_image_output = f32_visual_session.run(None, {"input": image_onnx})[0]
    f32onnx_text_output = f32_textual_session.run(None, {"input": text_onnx})[0]

    print(f"float16onnx image sum difference with normalization: {compute_dif(torch_image_output, f16onnx_image_output)}")

    print(f"float16onnx text sum difference with normalization: {compute_dif(torch_text_output ,f16onnx_text_output)}")

    print("---" * 20)

    print(f"float32onnx image sum difference with normalization: {compute_dif(torch_image_output, f32onnx_image_output)}")

    print(f"float32onnx text sum difference with normalization: {compute_dif(torch_text_output, f32onnx_text_output)}")




if __name__ == "__main__":
    SOURCE = "open_clip" #or "open_clip"
    MODEL_NAME= "ViT-L-14"
    PRETRAINED = "laion2b_s32b_b82k"  #only for open_clip

    onnx_evaluation(SOURCE, MODEL_NAME, PRETRAINED)

