"""
Lumon The Scrib v2 — Local Multimodal Translation Workstation.

Main Streamlit application with tabbed interface:
  📄 Text / Markdown — existing translation workflow (preserved)
  📑 Documents      — PDF, DOCX, image ingestion + translation
  📷 Camera         — live camera capture + OCR + translation
  📥 Export         — multi-format download
"""

import io
import time
import streamlit as st
import tiktoken

from lumon.config import (
    BACKENDS,
    SUPPORTED_LANGUAGES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_MAX_RESPONSE_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_TIMEOUT,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_TIMEOUT,
    TIKTOKEN_ENCODING,
)
from lumon.translation import (
    get_models,
    translate_with_retry,
)
from lumon.chunking import split_markdown, is_translatable
from lumon.glossary import (
    parse_manual_glossary,
    align_glossary_from_chunks,
)
from lumon.validation import validate_translation
from lumon.ocr import ocr_image_with_retry
from lumon.document import ingest, image_to_png_bytes
from lumon.export import EXPORT_FORMATS, EXPORT_EXTENSIONS
from lumon.camera import CaptureSession, CapturedPage, frame_to_bytes, create_thumbnail


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Lumon The Scrib",
    page_icon="📄",
    layout="wide",
)


# ──────────────────────────────────────────────
# Helper: make an OCR function bound to current settings
# ──────────────────────────────────────────────

def _make_ocr_func(base_url, ocr_model, backend_name, ocr_timeout):
    """Create a bound OCR function with current settings."""
    def _ocr(image_bytes: bytes, prompt_mode: str = "text") -> str:
        return ocr_image_with_retry(
            image_bytes=image_bytes,
            prompt_mode=prompt_mode,
            base_url=base_url,
            model=ocr_model,
            backend_name=backend_name,
            timeout=ocr_timeout,
        )
    return _ocr


# ──────────────────────────────────────────────
# Helper: run the translation pipeline on text
# ──────────────────────────────────────────────

def _run_translation(
    source_text: str,
    state_key: str,
    cache_key: str,
    glossary_key: str,
    backend_url: str,
    selected_model: str,
    src_lang: str,
    tgt_lang: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_response_tokens: int,
    request_timeout: int,
    backend_name: str,
    max_chunk_tokens: int,
    glossary_enabled: bool,
    manual_glossary_text: str,
):
    """
    Run the full translation pipeline on source text.
    Handles chunking, caching, glossary, progress, and validation.
    """
    start_time = time.time()
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    # Initialize glossary
    if glossary_key not in st.session_state:
        st.session_state[glossary_key] = {}
    auto_glossary: dict[str, str] = st.session_state[glossary_key]

    # Parse manual glossary
    manual_glossary = {}
    if manual_glossary_text.strip():
        manual_glossary = parse_manual_glossary(manual_glossary_text)

    # Merge: manual has priority over automatic
    combined_glossary = {**auto_glossary, **manual_glossary}

    # Initialize cache
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    cache: dict[int, str] = st.session_state[cache_key]

    # Chunking
    with st.spinner("Découpage du document..."):
        chunks = split_markdown(source_text, max_chunk_tokens)

    total_chunks = len(chunks)
    translatable = sum(1 for t, _ in chunks if is_translatable(t))
    cached_count = sum(1 for i, (t, _) in enumerate(chunks) if is_translatable(t) and i in cache)
    to_translate = translatable - cached_count

    info_msg = (
        f"Document découpé en **{total_chunks} blocs** "
        f"({translatable} à traduire, {total_chunks - translatable} conservés tels quels)"
    )
    if cached_count > 0:
        info_msg += f" — **{cached_count} blocs en cache**, {to_translate} restant(s)."
    if combined_glossary:
        info_msg += f" — 📖 {len(combined_glossary)} terme(s) dans le glossaire."
    st.info(info_msg)

    # Translation loop
    progress_bar = st.progress(0, text="Démarrage...")
    translated_parts: list[tuple[str, str]] = []
    error_occurred = False
    quality_warnings: list[str] = []

    for i, (chunk_type, chunk_text) in enumerate(chunks):
        progress_bar.progress(
            i / total_chunks,
            text=f"Bloc {i + 1}/{total_chunks}...",
        )

        # Non-translatable blocks
        if not is_translatable(chunk_type):
            translated_parts.append((chunk_type, chunk_text))
            continue

        # Check cache
        if i in cache:
            translated_parts.append(("text", cache[i]))
            continue

        # Prepare glossary
        chunk_glossary = combined_glossary if glossary_enabled else manual_glossary

        try:
            result = translate_with_retry(
                chunk_text, backend_url, selected_model,
                src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_response_tokens, request_timeout,
                backend_name,
                glossary=chunk_glossary if chunk_glossary else None,
            )
        except RuntimeError as e:
            error_occurred = True
            st.error(
                f"❌ Erreur au bloc {i + 1}/{total_chunks}. "
                f"**{len(cache)} blocs en cache** — recliquez pour reprendre.\n\n{e}"
            )
            break

        if result is None:
            error_occurred = True
            st.error(
                f"❌ Erreur au bloc {i + 1}/{total_chunks}. "
                f"**{len(cache)} blocs en cache** — recliquez pour reprendre."
            )
            break

        # Cache immediately
        cache[i] = result

        # Update automatic glossary
        if glossary_enabled:
            align_glossary_from_chunks(chunk_text, result, auto_glossary)
            combined_glossary = {**auto_glossary, **manual_glossary}

        # Validate quality
        chunk_warnings = validate_translation(chunk_text, result)
        for w in chunk_warnings:
            quality_warnings.append(f"Bloc {i + 1} : {w}")

        translated_parts.append(("text", result))

    if not error_occurred:
        progress_bar.progress(1.0, text="Terminé !")
        elapsed = time.time() - start_time

        result_text = "".join(content for _, content in translated_parts)
        st.session_state[state_key] = result_text
        st.success(
            f"Traduction terminée en **{elapsed:.1f}s** "
            f"({translatable} blocs traduits, "
            f"~{len(encoding.encode(result_text)):,} tokens)"
        )

        if quality_warnings:
            with st.expander(
                f"⚠️ {len(quality_warnings)} avertissement(s) qualité",
                expanded=False,
            ):
                for w in quality_warnings:
                    st.warning(w)


def _display_result(result_text: str, key_prefix: str, base_filename: str):
    """Display translation result with preview, raw view, and export options."""
    st.divider()

    tab_preview, tab_raw, tab_export = st.tabs([
        "📖 Aperçu Markdown", "📝 Texte brut", "📥 Exporter"
    ])

    with tab_preview:
        st.markdown(result_text)

    with tab_raw:
        st.text_area(
            "Résultat brut",
            value=result_text,
            height=400,
            label_visibility="collapsed",
            key=f"raw_{key_prefix}",
        )
        # Copy button via JavaScript
        import streamlit.components.v1 as components
        _escaped = result_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        components.html(
            f"""
            <button onclick="copyToClipboard()" id="copyBtn_{key_prefix}"
              style="
                background: #262730; color: #fafafa; border: 1px solid #4a4a5a;
                padding: 0.4rem 1rem; border-radius: 0.5rem; cursor: pointer;
                font-size: 0.85rem; transition: all 0.2s;
              "
              onmouseover="this.style.background='#3a3a4a'"
              onmouseout="this.style.background='#262730'"
            >📋 Copier le résultat brut</button>
            <script>
            function copyToClipboard() {{
                const text = `{_escaped}`;
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById('copyBtn_{key_prefix}');
                    btn.textContent = '✅ Copié !';
                    setTimeout(() => {{ btn.textContent = '📋 Copier le résultat brut'; }}, 2000);
                }}).catch(() => {{
                    const ta = document.createElement('textarea');
                    ta.value = text;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                    const btn = document.getElementById('copyBtn_{key_prefix}');
                    btn.textContent = '✅ Copié !';
                    setTimeout(() => {{ btn.textContent = '📋 Copier le résultat brut'; }}, 2000);
                }});
            }}
            </script>
            """,
            height=50,
        )

    with tab_export:
        st.subheader("📥 Exporter la traduction")

        export_col1, export_col2 = st.columns([2, 1])
        with export_col1:
            format_name = st.selectbox(
                "Format d'export",
                options=list(EXPORT_FORMATS.keys()),
                key=f"export_format_{key_prefix}",
            )
        with export_col2:
            ext = EXPORT_EXTENSIONS[format_name]
            export_filename = f"traduit_{base_filename}{ext}"

        # Generate export
        export_func = EXPORT_FORMATS[format_name]
        buf, fname, mime = export_func(result_text, export_filename)

        st.download_button(
            label=f"📥 Télécharger ({format_name})",
            data=buf,
            file_name=export_filename,
            mime=mime,
            use_container_width=True,
            key=f"dl_{key_prefix}_{format_name}",
        )


# ──────────────────────────────────────────────
# Sidebar Configuration
# ──────────────────────────────────────────────

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

    # Models
    st.divider()
    models = get_models(backend_url)

    # Translation model
    st.subheader("🔤 Modèle de traduction")
    if models:
        selected_model = st.selectbox("Modèle HY-MT", models, key="translation_model")
    else:
        st.warning("Aucun modèle détecté. Vérifiez que le serveur est lancé.")
        selected_model = st.text_input(
            "Nom du modèle (saisie manuelle)",
            help="Ex: HY-MT1.5-7B, HY-MT1.5-1.8B",
            key="translation_model_manual",
        )

    # OCR model
    st.subheader("👁️ Modèle OCR")
    if models:
        # Try to find a GLM-OCR model in the list
        ocr_default_idx = 0
        for i, m in enumerate(models):
            if "glm" in m.lower() and "ocr" in m.lower():
                ocr_default_idx = i
                break
        ocr_model = st.selectbox(
            "Modèle GLM-OCR",
            models,
            index=ocr_default_idx,
            key="ocr_model",
            help="Sélectionnez le modèle GLM-OCR pour la reconnaissance de texte.",
        )
    else:
        ocr_model = st.text_input(
            "Nom du modèle OCR",
            value=DEFAULT_OCR_MODEL,
            help="Ex: glm-ocr, glm-ocr:latest",
            key="ocr_model_manual",
        )

    # Languages
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

    # Advanced parameters (HY-MT recommended values)
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
    ocr_timeout = st.number_input(
        "Timeout OCR (s)",
        min_value=30, max_value=600,
        value=DEFAULT_OCR_TIMEOUT, step=30,
        help="Temps max d'attente pour l'OCR. L'OCR peut être plus lent.",
    )

    # Glossary
    st.divider()
    st.subheader("📖 Glossaire")

    glossary_enabled = st.toggle(
        "Glossaire automatique",
        value=True,
        help=(
            "Détecte automatiquement les noms propres et les injecte "
            "dans les prompts suivants pour assurer la cohérence."
        ),
    )
    manual_glossary_text = st.text_area(
        "Glossaire personnalisé",
        height=100,
        placeholder="Tokyo → 東京\nOpenAI → OpenAI\nMachine Learning -> Apprentissage automatique",
        help="Une entrée par ligne. Format : terme source → terme cible (ou ->).",
    )


# ──────────────────────────────────────────────
# Main Area
# ──────────────────────────────────────────────

st.title("📄 Lumon The Scrib")
st.caption(
    f"Traduction locale multimodale — "
    f"**{SUPPORTED_LANGUAGES[src_lang]}** → **{SUPPORTED_LANGUAGES[tgt_lang]}**"
)

# Create bound OCR function
ocr_func = _make_ocr_func(backend_url, ocr_model, backend_name, ocr_timeout)

# Tabs
tab_text, tab_docs, tab_camera, tab_about = st.tabs([
    "📄 Texte / Markdown",
    "📑 Documents",
    "📷 Caméra",
    "ℹ️ À propos",
])


# ──────────────────────────────────────────────
# TAB 1: Text / Markdown (existing workflow, preserved)
# ──────────────────────────────────────────────

with tab_text:
    st.subheader("📄 Traduction de texte / Markdown")

    # Input mode selection
    text_input_mode = st.radio(
        "Mode d'entrée",
        ["📁 Fichier (.md, .txt)", "✏️ Saisie directe"],
        horizontal=True,
        key="text_input_mode",
    )

    source_text = None
    file_identifier = "direct_input"

    if text_input_mode == "📁 Fichier (.md, .txt)":
        uploaded_file = st.file_uploader(
            "Choisissez un fichier",
            type=["md", "txt"],
            help="Fichiers Markdown (.md) ou texte (.txt)",
            key="text_file_upload",
        )
        if uploaded_file:
            source_text = uploaded_file.read().decode("utf-8")
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
    else:
        direct_text = st.text_area(
            "Texte source",
            height=300,
            placeholder="Collez votre texte ici...",
            key="direct_text_input",
        )
        if direct_text.strip():
            source_text = direct_text
            file_identifier = f"direct_{hash(direct_text) % 10**8}"

    if source_text and selected_model and src_lang != tgt_lang:
        encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        source_tokens = len(encoding.encode(source_text))

        col1, col2 = st.columns(2)
        col1.metric("Caractères", f"{len(source_text):,}")
        col2.metric("Tokens (est.)", f"{source_tokens:,}")

        st.caption(
            "⚠️ Estimation via **cl100k_base** (GPT-4). Le tokenizer HY-MT peut "
            "donner un comptage différent de ±15 %."
        )
        st.divider()

        state_key = f"result_{file_identifier}_{src_lang}_{tgt_lang}"
        cache_key = f"cache_{file_identifier}_{src_lang}_{tgt_lang}"
        glossary_key = f"glossary_{file_identifier}_{src_lang}_{tgt_lang}"

        if state_key not in st.session_state:
            st.session_state[state_key] = None
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}

        cache: dict[int, str] = st.session_state[cache_key]

        # Action buttons
        btn_col1, btn_col2 = st.columns([3, 1])
        launch = btn_col1.button(
            "🚀 Lancer la traduction", type="primary",
            use_container_width=True, key="launch_text",
        )
        if cache:
            if btn_col2.button("🗑️ Vider le cache", use_container_width=True, key="clear_cache_text"):
                st.session_state[cache_key] = {}
                st.session_state[state_key] = None
                st.rerun()

        if launch:
            _run_translation(
                source_text, state_key, cache_key, glossary_key,
                backend_url, selected_model, src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_response_tokens, request_timeout, backend_name,
                max_chunk_tokens, glossary_enabled, manual_glossary_text,
            )

        result_text = st.session_state.get(state_key)
        if result_text:
            base_name = file_identifier.split("_")[0] if "_" in file_identifier else "texte"
            _display_result(result_text, state_key, base_name)

    elif source_text is None:
        st.info("👆 Chargez un fichier ou saisissez du texte pour commencer.")
    elif src_lang == tgt_lang:
        pass
    elif not selected_model:
        st.warning("⚠️ Sélectionnez un modèle de traduction dans la barre latérale.")


# ──────────────────────────────────────────────
# TAB 2: Documents (PDF, DOCX, Images)
# ──────────────────────────────────────────────

with tab_docs:
    st.subheader("📑 Traduction de documents")
    st.caption("PDF, DOCX, images — extraction automatique + OCR si nécessaire")

    doc_file = st.file_uploader(
        "Choisissez un document",
        type=["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
        help="PDF, DOCX, ou fichiers image",
        key="doc_file_upload",
    )

    if doc_file:
        file_ext = doc_file.name.rsplit(".", 1)[-1].lower() if "." in doc_file.name else ""
        file_bytes = doc_file.read()

        st.metric("Fichier", doc_file.name)

        # OCR mode selection for images
        ocr_prompt_mode = "text"
        if file_ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"):
            ocr_prompt_mode = st.selectbox(
                "Mode OCR",
                options=["text", "formula", "table"],
                format_func=lambda m: {
                    "text": "📝 Texte général",
                    "formula": "🔢 Formules mathématiques",
                    "table": "📊 Tableaux",
                }[m],
                key="doc_ocr_mode",
            )

        st.divider()

        # State keys
        doc_state_key = f"doc_result_{doc_file.name}_{doc_file.size}_{src_lang}_{tgt_lang}"
        doc_extract_key = f"doc_extract_{doc_file.name}_{doc_file.size}"
        doc_cache_key = f"doc_cache_{doc_file.name}_{doc_file.size}_{src_lang}_{tgt_lang}"
        doc_glossary_key = f"doc_glossary_{doc_file.name}_{doc_file.size}_{src_lang}_{tgt_lang}"

        # Step 1: Extract text
        extract_col, translate_col = st.columns(2)

        with extract_col:
            if st.button("📤 Extraire le texte", use_container_width=True, key="extract_doc"):
                with st.spinner("Extraction en cours..."):

                    # Create OCR function bound to image prompt mode
                    def _doc_ocr(img_bytes, pm=ocr_prompt_mode):
                        return ocr_func(img_bytes, pm)

                    try:
                        progress_placeholder = st.empty()

                        def _progress(current, total, status=""):
                            progress_placeholder.progress(
                                current / max(total, 1),
                                text=status or f"Page {current}/{total}"
                            )

                        extracted = ingest(
                            file_bytes, file_ext, _doc_ocr,
                            progress_callback=_progress if file_ext == "pdf" else None,
                        )
                        st.session_state[doc_extract_key] = extracted
                        progress_placeholder.empty()
                        st.success(f"✅ Texte extrait ({len(extracted):,} caractères)")
                    except Exception as e:
                        st.error(f"❌ Erreur d'extraction : {e}")

        # Show extracted text
        extracted_text = st.session_state.get(doc_extract_key)
        if extracted_text:
            with st.expander("📋 Texte extrait (aperçu)", expanded=False):
                st.text_area(
                    "Texte extrait",
                    value=extracted_text[:5000] + ("..." if len(extracted_text) > 5000 else ""),
                    height=300,
                    label_visibility="collapsed",
                    key=f"preview_{doc_extract_key}",
                )

            # Step 2: Translate
            if selected_model and src_lang != tgt_lang:
                if doc_state_key not in st.session_state:
                    st.session_state[doc_state_key] = None

                launch_doc = st.button(
                    "🚀 Traduire le document", type="primary",
                    use_container_width=True, key="launch_doc",
                )

                if launch_doc:
                    _run_translation(
                        extracted_text, doc_state_key, doc_cache_key, doc_glossary_key,
                        backend_url, selected_model, src_lang, tgt_lang,
                        temperature, top_k, top_p, repetition_penalty,
                        max_response_tokens, request_timeout, backend_name,
                        max_chunk_tokens, glossary_enabled, manual_glossary_text,
                    )

                doc_result = st.session_state.get(doc_state_key)
                if doc_result:
                    base_name = doc_file.name.rsplit(".", 1)[0] if "." in doc_file.name else doc_file.name
                    _display_result(doc_result, doc_state_key, base_name)
    else:
        st.info("👆 Chargez un document PDF, DOCX ou une image pour commencer.")


# ──────────────────────────────────────────────
# TAB 3: Camera
# ──────────────────────────────────────────────

with tab_camera:
    st.subheader("📷 Traduction par caméra")
    st.caption("Capturez des pages avec votre caméra, OCR automatique + traduction")

    # Initialize capture session
    if "capture_session" not in st.session_state:
        st.session_state["capture_session"] = CaptureSession()
    session: CaptureSession = st.session_state["capture_session"]

    # Camera settings
    cam_col1, cam_col2 = st.columns([2, 1])

    with cam_col2:
        cam_ocr_mode = st.selectbox(
            "Mode OCR",
            options=["text", "formula", "table"],
            format_func=lambda m: {
                "text": "📝 Texte",
                "formula": "🔢 Formules",
                "table": "📊 Tableaux",
            }[m],
            key="cam_ocr_mode",
        )

        auto_interval = st.number_input(
            "Capture auto (secondes)",
            min_value=0, max_value=60, value=0, step=5,
            help="0 = désactivé. Capture automatique toutes les N secondes.",
            key="auto_interval",
        )
        session.auto_capture_interval = float(auto_interval)

    with cam_col1:
        # Camera input
        camera_image = st.camera_input(
            "📷 Capturer une page",
            key="camera_capture",
        )

    if camera_image:
        img_bytes = camera_image.getvalue()
        # Convert to PNG for consistent OCR
        png_bytes = image_to_png_bytes(img_bytes)

        # Add to session and run OCR
        page = session.add_page(png_bytes)

        with st.spinner(f"🔍 OCR page {page.page_number}..."):
            try:
                ocr_text = ocr_func(png_bytes, cam_ocr_mode)
                page.ocr_text = ocr_text
                st.success(f"✅ Page {page.page_number} — {len(ocr_text)} caractères reconnus")
            except Exception as e:
                st.error(f"❌ Erreur OCR : {e}")

    # Display captured pages
    if session.pages:
        st.divider()
        st.subheader(f"📚 Pages capturées ({len(session.pages)})")

        for idx, page in enumerate(session.pages):
            with st.expander(
                f"Page {page.page_number} — {len(page.ocr_text)} car.",
                expanded=(idx == len(session.pages) - 1),
            ):
                img_col, text_col = st.columns([1, 2])
                with img_col:
                    st.image(page.image_bytes, width=200)
                with text_col:
                    st.text_area(
                        f"OCR Page {page.page_number}",
                        value=page.ocr_text,
                        height=150,
                        label_visibility="collapsed",
                        key=f"ocr_page_{idx}",
                    )
                if st.button(f"🗑️ Supprimer", key=f"del_page_{idx}"):
                    session.remove_page(idx)
                    st.rerun()

        st.divider()

        # Combine all OCR text
        combined_ocr = session.get_all_ocr_text()

        if combined_ocr and selected_model and src_lang != tgt_lang:
            st.metric("Texte total reconnu", f"{len(combined_ocr):,} caractères")

            cam_state_key = f"cam_result_{hash(combined_ocr) % 10**8}_{src_lang}_{tgt_lang}"
            cam_cache_key = f"cam_cache_{hash(combined_ocr) % 10**8}_{src_lang}_{tgt_lang}"
            cam_glossary_key = f"cam_glossary_{src_lang}_{tgt_lang}"

            if cam_state_key not in st.session_state:
                st.session_state[cam_state_key] = None

            translate_cam = st.button(
                "🚀 Traduire toutes les pages", type="primary",
                use_container_width=True, key="launch_camera",
            )

            if translate_cam:
                _run_translation(
                    combined_ocr, cam_state_key, cam_cache_key, cam_glossary_key,
                    backend_url, selected_model, src_lang, tgt_lang,
                    temperature, top_k, top_p, repetition_penalty,
                    max_response_tokens, request_timeout, backend_name,
                    max_chunk_tokens, glossary_enabled, manual_glossary_text,
                )

            cam_result = st.session_state.get(cam_state_key)
            if cam_result:
                _display_result(cam_result, cam_state_key, "camera")

        # Clear session button
        if st.button("🗑️ Effacer toutes les captures", key="clear_session"):
            session.clear()
            st.rerun()

    else:
        st.info("👆 Utilisez la caméra ci-dessus pour capturer une page.")


# ──────────────────────────────────────────────
# TAB 4: About
# ──────────────────────────────────────────────

with tab_about:
    st.subheader("ℹ️ À propos de Lumon The Scrib")
    st.markdown("""
**Lumon The Scrib** est un outil de traduction local multimodal propulsé par :

- **[HY-MT](https://github.com/Tencent-Hunyuan/HY-MT)** — Modèle de traduction Hunyuan par Tencent (33 langues)
- **[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)** — Modèle OCR multimodal par Z.ai (0.9B paramètres)

### ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| **Traduction de texte** | Markdown, texte brut — découpage intelligent, glossaire, cache |
| **Documents** | PDF (digital + scanné), DOCX, images — extraction + OCR automatique |
| **Caméra** | Capture de pages, OCR progressif, traduction accumulée |
| **Export** | Markdown, TXT, PDF, HTML, DOCX |
| **100 % local** | Vos données ne quittent jamais votre machine |

### 🔧 Configuration requise

| Composant | Recommandation |
|---|---|
| **Python** | 3.10+ |
| **LM Studio** ou **Ollama** | Dernière version |
| **Modèle traduction** | HY-MT1.5-1.8B ou HY-MT1.5-7B |
| **Modèle OCR** | GLM-OCR |

### ⚙️ Paramètres HY-MT recommandés

Les valeurs par défaut de la sidebar correspondent aux **paramètres officiels HY-MT** :
température 0.7, top-k 20, top-p 0.6, repetition penalty 1.05.
    """)
