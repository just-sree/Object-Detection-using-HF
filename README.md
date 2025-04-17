# 🖼️ Object Detection & Scene Narration

[![HF Space](https://huggingface.co/spaces/just-sree/Object-Detection-and-Audio-Narration)

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
