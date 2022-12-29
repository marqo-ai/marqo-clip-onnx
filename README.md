This repo is used to generate the onnx version of the clip model.

check `main.py` for conversion

check `difference_test.py` for the performance of the onnx model in sum difference with normalization.

To use the scripts, specify the parameters:

`SOURCE` : either `openai` or `open_clip` .

`MODEL_NAME`: the name of the clip_model. Note that the model have different formats in `openai` and `open_clip`.

`PRETRAINED`: pretraied dataset. only needed for `open_clip` models.
