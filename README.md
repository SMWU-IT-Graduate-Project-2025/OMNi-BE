# OMNi-BE

<img src='https://i.ifh.cc/yHzGZf.png'><br>

```
conda create -n omni python=3.11 -y

conda activate omni

cd ./OMMi-BE

pip install -r requirements.txt

python main.py
```

---
### Export pretrained model from huggingface to onnx
```
optimum-cli export onnx --model Searchium-ai/clip4clip-webvid150k --task image-feature-extraction --opset 14 ./onnx_vision```