"""
Rendering helpers + NL summary
"""

import io
import logging
import warnings
from collections import OrderedDict

import inflect
import matplotlib.pyplot as plt
from PIL import Image
from transformers import logging as hf_logging


def render_results_in_image(img: Image.Image, preds):
    """Draw green boxes + red labels; return new PIL image."""
    img_w, img_h = img.size
    plt.figure(figsize=(min(16, img_w / 80), min(10, img_h / 80)))
    plt.imshow(img)
    ax = plt.gca()

    for p in preds:
        box = p["box"]
        # Convert normalized coords to pixel coords if needed
        if 0 <= box["xmin"] <= 1 and box["xmax"] <= 1:
            box["xmin"] *= img_w
            box["xmax"] *= img_w
            box["ymin"] *= img_h
            box["ymax"] *= img_h

        x, y = box["xmin"], box["ymin"]
        w, h = box["xmax"] - x, box["ymax"] - y
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="green", linewidth=2))
        ax.text(x, y, f"{p['label']} {p['score']:.2f}", color="red", fontsize=8)

    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    out = Image.open(buf)
    plt.close()
    return out


_inflect = inflect.engine()


def _plural(label, n):
    label = label if n == 1 else _inflect.plural_noun(label)
    return f"{_inflect.number_to_words(n)} {label}"


def summarize_predictions_natural_language(preds) -> str:
    """Oxford‑comma list: 'In this image, there are two cats and one dog.'"""
    if not preds:
        return "I couldn't detect any objects with high confidence."
    counter = OrderedDict()
    for p in preds:
        counter[p["label"]] = counter.get(p["label"], 0) + 1

    phrases = [_plural(lbl, n) for lbl, n in counter.items()]
    if len(phrases) > 1:
        summary = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    else:
        summary = phrases[0]
    return f"In this image, there are {summary}."


def ignore_warnings():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    hf_logging.set_verbosity_error()
