"""
Utility: convert text → MP3 via gTTS.
"""

from functools import lru_cache
import tempfile
from gtts import gTTS

SUPPORTED_TTS_LANGS = ["en", "fr", "de", "es", "pt", "it", "hi", "zh-CN"]

@lru_cache(maxsize=128)
def text_to_speech_mp3(text: str, lang: str = "en") -> str:
    if lang not in SUPPORTED_TTS_LANGS:
        lang = "en"
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name
