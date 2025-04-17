# 🖼️ Object Detection & Scene Narration

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20View‑on‑HuggingFace-Spaces-blue)](https://huggingface.co/spaces/<YOUR_USER>/<YOUR_SPACE>)

Upload a picture or paste an image URL →  
the app will **detect objects**, draw bounding boxes, describe what it sees, and – if you like – **speak** the summary aloud.

![demo GIF](assets/demo.gif)

## Features

* **DETR ResNet‑50** object detector (🤗 Transformers)  
* Confidence threshold slider  
* Natural‑language summary with correct pluralisation  
* Multilingual **gTTS** audio (English, French, Spanish, German, Portuguese, Italian, Hindi, Chinese)  
* Gradio interface – deploys instantly on HF Spaces  

## Run locally

```bash
git clone https://github.com/<YOU>/ObjectDetectionSpace
cd ObjectDetectionSpace
pip install -r requirements.txt
python app.py
