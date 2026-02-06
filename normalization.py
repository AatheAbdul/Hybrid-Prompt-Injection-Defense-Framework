import unicodedata
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import base64
import urllib.parse

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang != "en":
            text = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        pass  # fail-safe
    return text

def try_base64_decode(text: str) -> str:
    try:
        decoded = base64.b64decode(text).decode("utf-8")
        if len(decoded) > 0:
            return decoded
    except Exception:
        pass
    return text

def decode_url(text: str) -> str:
    return urllib.parse.unquote(text)

def normalize_encoding(text: str) -> str:
    text = decode_url(text)
    text = try_base64_decode(text)
    return text

def normalize_input(text: str) -> str:
    text = normalize_unicode(text)
    text = normalize_language(text)
    text = normalize_encoding(text)
    return text

