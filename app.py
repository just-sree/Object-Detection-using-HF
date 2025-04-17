"""
Object Detection & Scene Summary – Hugging Face Space
Enhanced version (2025‑04‑17)
"""

from functools import lru_cache
from pathlib import Path
from typing import Tuple, List

import gradio as gr
from PIL import Image
from transformers import pipeline
import requests
import torch
import helper
from tts import text_to_speech_mp3, SUPPORTED_TTS_LANGS


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
@lru_cache(maxsize=1)
def load_detector():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="object-detection",
        model="facebook/detr-resnet-50",
        device=device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Image + detection logic
def fetch_img_from_url(url: str) -> Image.Image:
    try:
        return Image.open(requests.get(url, stream=True, timeout=10).raw)
    except Exception as err:
        raise gr.Error(f"Could not retrieve image: {err}")


def run_detection(img: Image.Image, conf_thres: float) -> Tuple[Image.Image, List[dict]]:
    detector = load_detector()
    preds = detector(img)
    preds = [p for p in preds if p["score"] >= conf_thres]
    boxed_img = helper.render_results_in_image(img, preds)
    return boxed_img, preds


def pipeline_fn(
    img: Image.Image,
    img_url: str,
    conf_thres: float,
    want_audio: bool,
    tts_lang: str,
):
    if img is None and not img_url:
        raise gr.Error("Please upload an image or paste a URL.")
    if img is None and img_url:
        img = fetch_img_from_url(img_url)

    boxed_img, preds = run_detection(img, conf_thres)
    summary = helper.summarize_predictions_natural_language(preds)

    audio = None
    if want_audio and summary.strip():
        audio = text_to_speech_mp3(summary, tts_lang)  # ✅ Return file path only

    return boxed_img, summary, audio


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
title = "🖼️ Object Detection & Scene Narration"
description = (
    "Upload an image to detect objects, visualize boxes, and hear a summary. "
    "Powered by **facebook/detr‑resnet‑50**."
)

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload an image")
            img_url = gr.Textbox(label="…or paste image URL")
            conf = gr.Slider(minimum=0.05, maximum=0.9, step=0.05, value=0.5, label="Confidence threshold")
            want_audio = gr.Checkbox(label="Generate audio summary", value=False)
            tts_lang = gr.Dropdown(choices=SUPPORTED_TTS_LANGS, value="en", label="Audio language")
            run_btn = gr.Button("Detect ▶️")

        with gr.Column(scale=1):
            boxed_img = gr.Image(label="Detections", type="pil")
            summary = gr.Textbox(label="Scene description")
            audio_out = gr.Audio(label="Spoken summary (MP3)", type="filepath")  # ✅ Ensure Gradio knows this is a path

    run_btn.click(
        fn=pipeline_fn,
        inputs=[img_input, img_url, conf, want_audio, tts_lang],
        outputs=[boxed_img, summary, audio_out],
    )

    gr.Examples(examples=["sample.jpg"], inputs=img_input, label="Try an example")

if __name__ == "__main__":
    helper.ignore_warnings()
    demo.queue().launch()
