"""
Traducteur de documents Markdown via HY-MT (Hunyuan Translation Model).

Backend local : LM Studio ou Ollama servant le modèle HY-MT.
Optimisé pour les très gros documents :
- Prompts conformes à la documentation HY-MT (pas de system prompt)
- 33 langues supportées avec sélection source/cible
- Découpage intelligent préservant la structure Markdown
- Streaming des réponses pour éviter les timeouts
- Retry automatique avec backoff exponentiel
- Persistance du résultat via Streamlit session state
"""

import streamlit as st
import requests
import json
import re
import time
import tiktoken

# ──────────────────────────────────────────────
# Configuration par défaut
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
_ZH_TARGET_NAMES = {
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
_EN_TARGET_NAMES = {
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
# Construction du prompt HY-MT
# ──────────────────────────────────────────────

def _is_zh_involved(src_lang: str, tgt_lang: str) -> bool:
    """Vérifie si le chinois est impliqué dans la paire de traduction."""
    return src_lang in ("zh", "zh-Hant") or tgt_lang in ("zh", "zh-Hant")


def build_prompt(source_text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Construit le prompt conformément aux templates HY-MT.
    - ZH<=>XX : prompt en chinois
    - XX<=>XX (sans chinois) : prompt en anglais
    """
    if _is_zh_involved(src_lang, tgt_lang):
        target_name = _ZH_TARGET_NAMES.get(tgt_lang, tgt_lang)
        return (
            f"将以下文本翻译为{target_name}，"
            f"注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{source_text}"
        )
    else:
        target_name = _EN_TARGET_NAMES.get(tgt_lang, tgt_lang)
        return (
            f"Translate the following segment into {target_name}, "
            f"without additional explanation.\n\n"
            f"{source_text}"
        )


# ──────────────────────────────────────────────
# Fonctions API
# ──────────────────────────────────────────────

def get_models(base_url: str, timeout: int = 10) -> list[str]:
    """Récupère la liste des modèles disponibles via l'endpoint OpenAI-compatible."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data:
            return [m["id"] for m in data["data"]]
        return []
    except Exception:
        return []


def translate_chunk_stream(
    text: str,
    base_url: str,
    model: str,
    src_lang: str,
    tgt_lang: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    timeout: int,
) -> str:
    """
    Traduit un bloc de texte via l'API /v1/chat/completions en mode streaming.

    Conforme à HY-MT : pas de system prompt, prompt user uniquement.
    Paramètres d'inférence recommandés par la documentation.

    Le décodage UTF-8 est forcé manuellement pour éviter les artefacts
    d'encodage (ex. 脙漏 au lieu de é) causés par le décodage ISO-8859-1
    par défaut de la bibliothèque requests.
    """
    prompt = build_prompt(text, src_lang, tgt_lang)

    # HY-MT n'utilise pas de system prompt par défaut
    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "stream": True,
    }

    # En-têtes explicites pour forcer UTF-8 côté serveur et côté client
    headers = {
        "Accept": "application/json; charset=utf-8",
        "Content-Type": "application/json; charset=utf-8",
    }

    collected: list[str] = []
    with requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        # ensure_ascii=False préserve les caractères non-ASCII dans le JSON
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        # decode_unicode=False : on reçoit des bytes bruts
        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            # Décodage UTF-8 explicite (replace pour ne jamais planter)
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    collected.append(content)
            except (ValueError, KeyError, IndexError) as e:
                print(f"[UTF-8 DEBUG] Parse error: {e}, raw: {raw_line!r}")
                continue

    return "".join(collected)


def translate_with_retry(
    text: str,
    base_url: str,
    model: str,
    src_lang: str,
    tgt_lang: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    timeout: int,
    max_retries: int = MAX_RETRIES,
) -> str | None:
    """Tente la traduction avec retry et backoff exponentiel."""
    for attempt in range(max_retries):
        try:
            return translate_chunk_stream(
                text, base_url, model, src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_tokens, timeout,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                st.error(f"Échec après {max_retries} tentatives : {e}")
                return None


# ──────────────────────────────────────────────
# Découpage intelligent du Markdown
# ──────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def _split_preserving_blocks(text: str) -> list[str]:
    """
    Découpe le texte en segments en préservant les blocs spéciaux
    (code fences, front matter) comme unités atomiques.
    """
    segments: list[str] = []

    # Extraire le front matter s'il existe
    fm_match = _FRONT_MATTER_RE.match(text)
    if fm_match:
        segments.append(fm_match.group(0))
        text = text[fm_match.end():]

    # Découper autour des code fences
    parts = re.split(
        r"(^`{3,}.*?^`{3,}.*?$|^~{3,}.*?^~{3,}.*?$)",
        text, flags=re.MULTILINE | re.DOTALL,
    )

    for part in parts:
        if not part:
            continue
        if _CODE_FENCE_RE.match(part):
            # Bloc de code → unité atomique (ne pas traduire)
            segments.append(part)
        else:
            # Découper par double saut de ligne (paragraphes)
            paragraphs = re.split(r"(\n\s*\n)", part)
            segments.extend(p for p in paragraphs if p)

    return segments


def split_markdown(text: str, max_tokens: int) -> list[str]:
    """
    Découpe le document Markdown en chunks respectant la limite de tokens,
    tout en préservant les blocs de code et le front matter.
    Les blocs de code sont conservés tels quels (non traduits).
    """
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    segments = _split_preserving_blocks(text)

    chunks: list[str] = []
    current_chunk = ""
    current_tokens = 0

    for segment in segments:
        seg_tokens = len(encoding.encode(segment))

        if seg_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0

            lines = segment.splitlines(keepends=True)
            for line in lines:
                line_tokens = len(encoding.encode(line))
                if current_tokens + line_tokens <= max_tokens:
                    current_chunk += line
                    current_tokens += line_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line
                    current_tokens = line_tokens
        elif current_tokens + seg_tokens <= max_tokens:
            current_chunk += segment
            current_tokens += seg_tokens
        else:
            chunks.append(current_chunk)
            current_chunk = segment
            current_tokens = seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def is_code_block(text: str) -> bool:
    """Vérifie si un chunk est un bloc de code (ne doit pas être traduit)."""
    stripped = text.strip()
    return bool(_CODE_FENCE_RE.match(stripped))


def is_front_matter(text: str) -> bool:
    """Vérifie si un chunk est du front matter YAML (ne doit pas être traduit)."""
    return text.strip().startswith("---") and text.strip().endswith("---")


# ──────────────────────────────────────────────
# Interface Streamlit
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Lumon The Scrib",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar : configuration ──

with st.sidebar:
    st.header("⚙️ Configuration")

    # Backend
    backend_name = st.selectbox(
        "Backend LLM",
        options=list(BACKENDS.keys()),
        format_func=lambda b: f"{BACKENDS[b]['icon']} {b}",
    )
    backend_url = st.text_input(
        "URL du serveur",
        value=BACKENDS[backend_name]["url"],
        help="Modifiez si votre serveur utilise un port différent.",
    )

    # Modèles
    st.divider()
    models = get_models(backend_url)
    if models:
        selected_model = st.selectbox("Modèle", models)
    else:
        st.warning("Aucun modèle détecté. Vérifiez que le serveur est lancé.")
        selected_model = st.text_input(
            "Nom du modèle (saisie manuelle)",
            help="Ex: HY-MT1.5-7B, HY-MT1.5-1.8B",
        )

    # Langues
    st.divider()
    st.subheader("🌍 Langues")
    lang_codes = list(SUPPORTED_LANGUAGES.keys())
    lang_labels = list(SUPPORTED_LANGUAGES.values())

    src_lang_idx = st.selectbox(
        "Langue source",
        range(len(lang_codes)),
        format_func=lambda i: lang_labels[i],
        index=lang_codes.index("en"),
    )
    tgt_lang_idx = st.selectbox(
        "Langue cible",
        range(len(lang_codes)),
        format_func=lambda i: lang_labels[i],
        index=lang_codes.index("fr"),
    )
    src_lang = lang_codes[src_lang_idx]
    tgt_lang = lang_codes[tgt_lang_idx]

    if src_lang == tgt_lang:
        st.error("⚠️ La langue source et cible doivent être différentes.")

    # Paramètres avancés (valeurs recommandées par HY-MT)
    st.divider()
    st.subheader("🔧 Paramètres avancés")

    max_chunk_tokens = st.number_input(
        "Tokens max par bloc",
        min_value=100, max_value=8000,
        value=DEFAULT_CHUNK_TOKENS, step=100,
        help="Taille maximale de chaque bloc envoyé au modèle.",
    )
    temperature = st.slider(
        "Température",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_TEMPERATURE, step=0.05,
        help="Recommandé par HY-MT : 0.7",
    )
    top_k = st.number_input(
        "Top-K",
        min_value=1, max_value=100,
        value=DEFAULT_TOP_K, step=1,
        help="Recommandé par HY-MT : 20",
    )
    top_p = st.slider(
        "Top-P",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_TOP_P, step=0.05,
        help="Recommandé par HY-MT : 0.6",
    )
    repetition_penalty = st.slider(
        "Repetition penalty",
        min_value=1.0, max_value=2.0,
        value=DEFAULT_REPETITION_PENALTY, step=0.05,
        help="Recommandé par HY-MT : 1.05",
    )
    max_response_tokens = st.number_input(
        "Tokens max par réponse",
        min_value=100, max_value=8000,
        value=DEFAULT_MAX_RESPONSE_TOKENS, step=100,
    )
    request_timeout = st.number_input(
        "Timeout par requête (s)",
        min_value=30, max_value=600,
        value=DEFAULT_TIMEOUT, step=30,
        help="Temps max d'attente par bloc. Augmentez pour les très gros blocs.",
    )

# ── Zone principale ──

st.title("📄 Lumon The Scrib")
st.caption(
    f"Traduction locale de gros documents Markdown — "
    f"**{SUPPORTED_LANGUAGES[src_lang]}** → **{SUPPORTED_LANGUAGES[tgt_lang]}**"
)

# Upload
uploaded_file = st.file_uploader(
    "Choisissez un fichier Markdown",
    type=["md"],
    help="Seuls les fichiers .md sont acceptés.",
)

if uploaded_file and selected_model and src_lang != tgt_lang:
    source_text = uploaded_file.read().decode("utf-8")
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    source_tokens = len(encoding.encode(source_text))

    # Statistiques du fichier
    col1, col2, col3 = st.columns(3)
    col1.metric("Fichier", uploaded_file.name)
    col2.metric("Caractères", f"{len(source_text):,}")
    col3.metric("Tokens (est.)", f"{source_tokens:,}")

    st.divider()

    # Session state
    state_key = f"result_{uploaded_file.name}_{uploaded_file.size}_{src_lang}_{tgt_lang}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    # Bouton de traduction
    if st.button("🚀 Lancer la traduction", type="primary", use_container_width=True):
        start_time = time.time()

        # Découpage
        with st.spinner("Découpage du document..."):
            chunks = split_markdown(source_text, max_chunk_tokens)

        total_chunks = len(chunks)
        translatable = sum(1 for c in chunks if not is_code_block(c) and not is_front_matter(c))
        st.info(
            f"Document découpé en **{total_chunks} blocs** "
            f"({translatable} à traduire, {total_chunks - translatable} conservés tels quels)"
        )

        # Traduction
        progress_bar = st.progress(0, text="Démarrage...")
        translated_chunks: list[str] = []
        error_occurred = False

        for i, chunk in enumerate(chunks):
            progress_bar.progress(
                i / total_chunks,
                text=f"Bloc {i + 1}/{total_chunks}...",
            )

            # Ne pas traduire les blocs de code et le front matter
            if is_code_block(chunk) or is_front_matter(chunk):
                translated_chunks.append(chunk)
                continue

            result = translate_with_retry(
                chunk, backend_url, selected_model,
                src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_response_tokens, request_timeout,
            )
            if result is None:
                error_occurred = True
                st.error(f"Erreur au bloc {i + 1}/{total_chunks}. Traduction interrompue.")
                break
            translated_chunks.append(result)

        if not error_occurred:
            progress_bar.progress(1.0, text="Terminé !")
            elapsed = time.time() - start_time
            result_text = "\n\n".join(translated_chunks)

            st.session_state[state_key] = result_text
            st.success(
                f"Traduction terminée en **{elapsed:.1f}s** "
                f"({translatable} blocs traduits, "
                f"~{len(encoding.encode(result_text)):,} tokens)"
            )

    # Affichage du résultat
    result_text = st.session_state.get(state_key)
    if result_text:
        st.divider()

        tab_preview, tab_raw = st.tabs(["📖 Aperçu Markdown", "📝 Texte brut"])
        with tab_preview:
            st.markdown(result_text)
        with tab_raw:
            st.code(result_text, language="markdown")

        st.download_button(
            label="📥 Télécharger le fichier traduit",
            data=result_text.encode("utf-8"),
            file_name=f"traduit_{uploaded_file.name}",
            mime="text/markdown",
            use_container_width=True,
        )

elif not uploaded_file:
    st.info("👆 Chargez un fichier `.md` pour commencer.")
elif src_lang == tgt_lang:
    pass  # Message d'erreur déjà affiché dans la sidebar
elif not selected_model:
    st.warning("⚠️ Sélectionnez ou saisissez un modèle dans la barre latérale.")