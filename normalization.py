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
    # Pattern to find potential Base64 strings: 8+ chars of A-Za-z0-9+/=
    b64_pattern = re.compile(r'[A-Za-z0-9+/=]{8,}')
    
    def decode_repl(match):
        candidate = match.group(0)
        try:
            # Basic validation: length should be multiple of 4
            if len(candidate) % 4 != 0:
                return candidate
            decoded = base64.b64decode(candidate).decode("utf-8")
            # If it's alphanumeric/readable, replace it
            if any(c.isprintable() for c in decoded):
                return f"[DECODED: {decoded}]"
        except Exception:
            pass
        return candidate

    return b64_pattern.sub(decode_repl, text)

def decode_url(text: str) -> str:
    return urllib.parse.unquote(text)

def normalize_encoding(text: str) -> str:
    text = decode_url(text)
    text = try_base64_decode(text)
    return text

def normalize_input(text: str) -> str:
    text = normalize_unicode(text)
    # 1. Decode FIRST (uncover hidden payload)
    text = normalize_encoding(text)
    # 2. Translate SECOND (translate both outer text and decoded payload)
    text = normalize_language(text)
    return text

