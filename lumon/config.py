"""
Configuration constants and language maps.

Extracted verbatim from traduction_app.py — all values preserved exactly.
"""

# ──────────────────────────────────────────────
# Backend configuration
# ──────────────────────────────────────────────

BACKENDS = {
    "LM Studio": {"url": "http://localhost:1234", "icon": "🖥️"},
    "Ollama": {"url": "http://localhost:11434", "icon": "🦙"},
}

# Paramètres d'inférence recommandés par HY-MT
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.6
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_RESPONSE_TOKENS = 2048
DEFAULT_CHUNK_TOKENS = 1500
DEFAULT_TIMEOUT = 120  # secondes
MAX_RETRIES = 3
TIKTOKEN_ENCODING = "cl100k_base"

# ──────────────────────────────────────────────
# Langues supportées par HY-MT
# ──────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "zh": "Chinese (中文)",
    "en": "English",
    "fr": "French (Français)",
    "pt": "Portuguese (Português)",
    "es": "Spanish (Español)",
    "ja": "Japanese (日本語)",
    "tr": "Turkish (Türkçe)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "ko": "Korean (한국어)",
    "th": "Thai (ภาษาไทย)",
    "it": "Italian (Italiano)",
    "de": "German (Deutsch)",
    "vi": "Vietnamese (Tiếng Việt)",
    "ms": "Malay (Bahasa Melayu)",
    "id": "Indonesian (Bahasa Indonesia)",
    "tl": "Filipino",
    "hi": "Hindi (हिन्दी)",
    "zh-Hant": "Traditional Chinese (繁體中文)",
    "pl": "Polish (Polski)",
    "cs": "Czech (Čeština)",
    "nl": "Dutch (Nederlands)",
    "km": "Khmer (ខ្មែរ)",
    "my": "Burmese (ဗမာ)",
    "fa": "Persian (فارسی)",
    "gu": "Gujarati (ગુજરાતી)",
    "ur": "Urdu (اردو)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "he": "Hebrew (עברית)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "uk": "Ukrainian (Українська)",
    "bo": "Tibetan (བོད་སྐད)",
    "kk": "Kazakh (Қазақ)",
    "mn": "Mongolian (Монгол)",
    "ug": "Uyghur (ئۇيغۇرچە)",
    "yue": "Cantonese (粤語)",
}

# Langues dont le nom cible doit être en chinois (pour prompt ZH<=>XX)
ZH_TARGET_NAMES = {
    "zh": "中文", "en": "英语", "fr": "法语", "pt": "葡萄牙语",
    "es": "西班牙语", "ja": "日语", "tr": "土耳其语", "ru": "俄语",
    "ar": "阿拉伯语", "ko": "韩语", "th": "泰语", "it": "意大利语",
    "de": "德语", "vi": "越南语", "ms": "马来语", "id": "印尼语",
    "tl": "菲律宾语", "hi": "印地语", "zh-Hant": "繁体中文",
    "pl": "波兰语", "cs": "捷克语", "nl": "荷兰语", "km": "高棉语",
    "my": "缅甸语", "fa": "波斯语", "gu": "古吉拉特语", "ur": "乌尔都语",
    "te": "泰卢固语", "mr": "马拉地语", "he": "希伯来语", "bn": "孟加拉语",
    "ta": "泰米尔语", "uk": "乌克兰语", "bo": "藏语", "kk": "哈萨克语",
    "mn": "蒙古语", "ug": "维吾尔语", "yue": "粤语",
}

# Noms de langues en anglais (pour prompt XX<=>XX hors chinois)
EN_TARGET_NAMES = {
    "zh": "Chinese", "en": "English", "fr": "French", "pt": "Portuguese",
    "es": "Spanish", "ja": "Japanese", "tr": "Turkish", "ru": "Russian",
    "ar": "Arabic", "ko": "Korean", "th": "Thai", "it": "Italian",
    "de": "German", "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian",
    "tl": "Filipino", "hi": "Hindi", "zh-Hant": "Traditional Chinese",
    "pl": "Polish", "cs": "Czech", "nl": "Dutch", "km": "Khmer",
    "my": "Burmese", "fa": "Persian", "gu": "Gujarati", "ur": "Urdu",
    "te": "Telugu", "mr": "Marathi", "he": "Hebrew", "bn": "Bengali",
    "ta": "Tamil", "uk": "Ukrainian", "bo": "Tibetan", "kk": "Kazakh",
    "mn": "Mongolian", "ug": "Uyghur", "yue": "Cantonese",
}

# ──────────────────────────────────────────────
# OCR Configuration (GLM-OCR)
# ──────────────────────────────────────────────

OCR_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}

DEFAULT_OCR_MODEL = "glm-ocr"
DEFAULT_OCR_TEMPERATURE = 0.01
DEFAULT_OCR_MAX_TOKENS = 8192
DEFAULT_OCR_TIMEOUT = 300  # secondes — OCR can be slower
