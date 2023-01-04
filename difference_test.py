import open_clip
import numpy as np
import torch
import clip
import onnx
import onnxruntime as ort
from PIL import Image
from huggingface_hub import login
from huggingface_hub import HfApi
import os
import shutil


def compute_dif(a, b):
    return np.abs(np.array(a) / np.linalg.norm(a) - np.array(np.array(b) / np.linalg.norm(b))).sum()


def onnx_evaluation(SOURCE, MODEL_NAME, PRETRAINED):
    ONNX16_VISUAL_DIR = "./onnx16-visual"
    ONNX16_TEXTUAL_DIR = "./onnx16-textual"

    ONNX32_VISUAL_DIR = "./onnx32-visual"
    ONNX32_TEXTUAL_DIR = "./onnx32-textual"

    dir_list = [ONNX16_VISUAL_DIR, ONNX16_TEXTUAL_DIR, ONNX32_VISUAL_DIR, ONNX32_TEXTUAL_DIR]
    for dir in dir_list:
        if os.path.isdir(dir) is False:
            os.mkdir(dir)

    if SOURCE == "openai":

        f32_VISUAL_PATH = os.path.join(ONNX32_VISUAL_DIR, f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx")
        f32_TEXTUAL_PATH = os.path.join(ONNX32_TEXTUAL_DIR,
                                        f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx")

        f16_VISUAL_PATH = os.path.join(ONNX16_VISUAL_DIR, f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx")
        f16_TEXTUAL_PATH = os.path.join(ONNX16_TEXTUAL_DIR,
                                        f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx")

        # model, preprocess = clip.load(name = MODEL_NAME, device = "cpu", jit=False)
        # tokenizer = clip.tokenize

    elif SOURCE == "open_clip":
        f32_VISUAL_PATH = os.path.join(ONNX32_VISUAL_DIR, f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx")
        f32_TEXTUAL_PATH = os.path.join(ONNX32_TEXTUAL_DIR,
                                        f"onnx32-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx")

        f16_VISUAL_PATH = os.path.join(ONNX16_VISUAL_DIR, f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-visual.onnx")
        f16_TEXTUAL_PATH = os.path.join(ONNX16_TEXTUAL_DIR,
                                        f"onnx16-{SOURCE}-{MODEL_NAME.replace('/', '-')}-textual.onnx")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name=MODEL_NAME, device="cpu",
                                                                     pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    f16_textual_session = ort.InferenceSession(f16_TEXTUAL_PATH,
                                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    f16_visual_session = ort.InferenceSession(f16_VISUAL_PATH,
                                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    f32_textual_session = ort.InferenceSession(f32_TEXTUAL_PATH,
                                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    f32_visual_session = ort.InferenceSession(f32_VISUAL_PATH,
                                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

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

    print(
        f"float16onnx image sum difference with normalization: {compute_dif(torch_image_output, f16onnx_image_output)}")

    print(f"float16onnx text sum difference with normalization: {compute_dif(torch_text_output, f16onnx_text_output)}")

    print("---" * 20)

    print(
        f"float32onnx image sum difference with normalization: {compute_dif(torch_image_output, f32onnx_image_output)}")

    print(f"float32onnx text sum difference with normalization: {compute_dif(torch_text_output, f32onnx_text_output)}")

    onnx16_dict = {
        f"onnx16/open_clip/{MODEL_NAME}/{PRETRAINED}":
            {
                "name": f"onnx16/open_clip/{MODEL_NAME}/{PRETRAINED}",
                "dimensions": np.array(f32onnx_image_output).shape[-1],
                "type": "clip_onnx",
                "note": f"the onnx float16 version of open_clip {MODEL_NAME}/{PRETRAINED}",
                "repo_id": f"Marqo/onnx-open_clip-{MODEL_NAME}",
                "visual_file": f"{os.path.basename(f16_VISUAL_PATH)}",
                "textual_file": f"{os.path.basename(f16_TEXTUAL_PATH)}",
                "token": None,
                "resolution": f32_visual_session.get_inputs()[0].shape[-1],
                "pretrained": f"{PRETRAINED}",
                "image_mean": None,
                "image_std": None,
            },
    }



    onnx32_dict = {
        f"onnx32/open_clip/{MODEL_NAME}/{PRETRAINED}":
            {
                "name": f"onnx32/open_clip/{MODEL_NAME}/{PRETRAINED}",
                "dimensions": np.array(f32onnx_image_output).shape[-1],
                "type": "clip_onnx",
                "note": f"the onnx float32 version of open_clip {MODEL_NAME}/{PRETRAINED}",
                "repo_id": f"Marqo/onnx-open_clip-{MODEL_NAME}",
                "visual_file": f"{os.path.basename(f32_VISUAL_PATH)}",
                "textual_file": f"{os.path.basename(f32_TEXTUAL_PATH)}",
                "token": None,
                "resolution": f32_visual_session.get_inputs()[0].shape[-1],
                "pretrained": f"{PRETRAINED}",
                "image_mean": None,
                "image_std": None,
            },
    }



    login("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")

    api = HfApi()

    model_path_list = [f32_VISUAL_PATH, f32_TEXTUAL_PATH, f16_VISUAL_PATH, f16_TEXTUAL_PATH]
    for model_path in model_path_list:
        dir = os.path.dirname(model_path)
    if len(os.listdir(dir)) == 1 and (os.path.basename(model_path) in os.listdir(dir)):
        MODEL_FILE_NAME = os.path.basename(model_path)

    elif len(os.listdir(dir)) > 1 and (os.path.basename(model_path) in os.listdir(dir)):
        MODEL_FILE_NAME = os.path.basename(model_path).replace(".onnx", ".zip")
        model_path = shutil.make_archive(MODEL_FILE_NAME.replace(".zip", ""), "zip", dir)
        onnx32_dict["visual_file"] = onnx32_dict["visual_file"].replace(".onnx", ".zip")

    api.upload_file(
    path_or_fileobj = model_path,
    path_in_repo = MODEL_FILE_NAME,
    repo_id = "Marqo/onnx-" + "open_clip-" + MODEL_NAME,
    repo_type = "model")





if __name__ == "__main__":
    SOURCE = "open_clip"  # or "open_clip"
MODEL_NAME = "ViT-H-14"
PRETRAINED = 'laion2b_s32b_b79k'  # only for open_clip

onnx_evaluation(SOURCE, MODEL_NAME, PRETRAINED)

