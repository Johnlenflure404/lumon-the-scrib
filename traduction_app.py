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
# Extraction heuristique de noms propres (Fix #7)
# ──────────────────────────────────────────────

# Regex pour noms propres latins : mot(s) commençant par majuscule
# Accepte : NomSimple, Nom Composé, OpenAI, NASA, GPT-4
_PROPER_NOUN_RE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-Þà-ÿ\-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-Þà-ÿ\-]*)*)\b"
)

# Mots courants à ignorer (articles, prépositions, etc.)
_STOP_WORDS = {
    # Anglais
    "The", "This", "That", "These", "Those", "There", "Then", "They",
    "Their", "Here", "Where", "When", "What", "Which", "With", "From",
    "Into", "About", "After", "Before", "During", "Between", "Under",
    "Above", "Below", "Each", "Every", "Some", "Many", "Much", "Most",
    "Other", "Another", "Such", "Only", "Also", "Just", "Very", "More",
    "Less", "However", "Therefore", "Furthermore", "Moreover", "Although",
    "Because", "Since", "While", "Until", "Unless", "Though", "Still",
    "Yet", "But", "And", "Not", "For", "All", "Any", "How", "Why",
    "Are", "Was", "Were", "Been", "Being", "Have", "Has", "Had",
    "Will", "Would", "Could", "Should", "May", "Might", "Must",
    "Can", "Shall", "Does", "Did", "Its", "Our", "His", "Her",
    # Français
    "Les", "Des", "Une", "Aux", "Par", "Sur", "Dans", "Pour",
    "Avec", "Sans", "Sous", "Vers", "Chez", "Entre", "Comme",
    "Mais", "Donc", "Car", "Puis", "Ici", "Cet", "Cette",
    "Sont", "Est", "Ont", "Qui", "Que", "Quoi",
    # Communs
    "Note", "See", "New", "Old", "Table", "Figure", "Section",
    "Chapter", "Part", "Step", "Example", "Data", "Type",
}


def extract_proper_nouns(text: str) -> set[str]:
    """
    Extrait les noms propres probables d'un texte latin.
    Ignore les mots en début de phrase et les stop words.
    """
    candidates: set[str] = set()
    # Découper en phrases pour ignorer le premier mot de chaque phrase
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 2:
            continue
        # Chercher dans toute la phrase sauf le premier mot
        search_text = " ".join(words[1:])
        for match in _PROPER_NOUN_RE.finditer(search_text):
            candidate = match.group(1)
            # Filtrer les stop words et les mots trop courts
            if candidate not in _STOP_WORDS and len(candidate) > 1:
                candidates.add(candidate)
    return candidates


def align_glossary_from_chunks(
    source_text: str,
    translated_text: str,
    existing_glossary: dict[str, str],
) -> dict[str, str]:
    """
    Met à jour le glossaire en détectant les noms propres dans le source
    et en les alignant avec la traduction.

    Heuristique : si un nom propre du source apparaît tel quel dans la
    traduction, on le conserve (translittération). Sinon, on ne l'ajoute
    pas automatiquement (trop risqué sans alignement mot-à-mot).
    """
    source_nouns = extract_proper_nouns(source_text)
    for noun in source_nouns:
        if noun in existing_glossary:
            continue
        # Si le nom apparaît tel quel dans la traduction → conservé
        if noun in translated_text:
            existing_glossary[noun] = noun
        else:
            # Chercher une version potentiellement différente
            # (ex: "Tokyo" → "東京"). Pour l'instant on ne fait pas
            # d'alignement complexe — le glossaire manuel couvre ce cas.
            pass
    return existing_glossary


def parse_manual_glossary(text: str) -> dict[str, str]:
    """
    Parse un glossaire saisi manuellement.
    Format attendu : une entrée par ligne, séparée par → ou ->
    Ex: "Tokyo → 東京"
    """
    glossary: dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Accepter → ou ->
        for sep in ("→", "->"):
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    src = parts[0].strip()
                    tgt = parts[1].strip()
                    if src and tgt:
                        glossary[src] = tgt
                break
    return glossary


# ──────────────────────────────────────────────
# Construction du prompt HY-MT
# ──────────────────────────────────────────────

def _is_zh_involved(src_lang: str, tgt_lang: str) -> bool:
    """Vérifie si le chinois est impliqué dans la paire de traduction."""
    return src_lang in ("zh", "zh-Hant") or tgt_lang in ("zh", "zh-Hant")


def _filter_glossary_for_chunk(
    glossary: dict[str, str], source_text: str,
) -> dict[str, str]:
    """Ne garde que les entrées du glossaire présentes dans le chunk source."""
    return {
        src: tgt for src, tgt in glossary.items()
        if src in source_text
    }


def build_prompt(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    glossary: dict[str, str] | None = None,
) -> str:
    """
    Construit le prompt conformément aux templates HY-MT.
    - ZH<=>XX : prompt en chinois
    - XX<=>XX (sans chinois) : prompt en anglais
    - Si glossaire fourni : utilise le template d'intervention terminologique
    """
    # Filtrer le glossaire pour ne garder que les termes du chunk
    active_glossary = {}
    if glossary:
        active_glossary = _filter_glossary_for_chunk(glossary, source_text)

    if _is_zh_involved(src_lang, tgt_lang):
        target_name = _ZH_TARGET_NAMES.get(tgt_lang, tgt_lang)
        if active_glossary:
            # Template terminologique HY-MT (doc officielle)
            entries = "\n".join(
                f"{src} 翻译成 {tgt}" for src, tgt in active_glossary.items()
            )
            return (
                f"参考下面的翻译：\n{entries}\n\n"
                f"将以下文本翻译为{target_name}，"
                f"注意只需要输出翻译后的结果，不要额外解释：\n"
                f"{source_text}"
            )
        return (
            f"将以下文本翻译为{target_name}，"
            f"注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{source_text}"
        )
    else:
        target_name = _EN_TARGET_NAMES.get(tgt_lang, tgt_lang)
        if active_glossary:
            entries = "\n".join(
                f"{src} → {tgt}" for src, tgt in active_glossary.items()
            )
            return (
                f"Refer to the following translations:\n{entries}\n\n"
                f"Translate the following segment into {target_name}, "
                f"without additional explanation.\n\n"
                f"{source_text}"
            )
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
    backend_name: str = "LM Studio",
    glossary: dict[str, str] | None = None,
) -> str:
    """
    Traduit un bloc de texte via l'API /v1/chat/completions en mode streaming.

    Conforme à HY-MT : pas de system prompt, prompt user uniquement.
    Paramètres d'inférence recommandés par la documentation.

    Le décodage UTF-8 est forcé manuellement pour éviter les artefacts
    d'encodage (ex. 脙漏 au lieu de é) causés par le décodage ISO-8859-1
    par défaut de la bibliothèque requests.

    Le payload est adapté selon le backend :
    - LM Studio : top_k et repetition_penalty en racine
    - Ollama : top_k et repeat_penalty dans un objet "options"
    """
    prompt = build_prompt(text, src_lang, tgt_lang, glossary)

    # HY-MT n'utilise pas de system prompt par défaut
    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }

    # ── Fix #6 : Adapter le payload selon le backend ──
    if backend_name == "Ollama":
        # Ollama attend top_k et repeat_penalty dans "options"
        payload["options"] = {
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
        }
    else:
        # LM Studio / OpenAI-compatible : paramètres en racine
        payload["top_k"] = top_k
        payload["repetition_penalty"] = repetition_penalty

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
    backend_name: str = "LM Studio",
    glossary: dict[str, str] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str | None:
    """Tente la traduction avec retry et backoff exponentiel."""
    for attempt in range(max_retries):
        try:
            return translate_chunk_stream(
                text, base_url, model, src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_tokens, timeout, backend_name, glossary,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                st.error(f"Échec après {max_retries} tentatives : {e}")
                return None


# ──────────────────────────────────────────────
# Découpage intelligent du Markdown (Fix #1 + #2)
# ──────────────────────────────────────────────
# State machine remplaçant la regex fragile.
# Chaque segment est un tuple (type, contenu) :
#   - ("code", "```py\nprint()\n```")
#   - ("front_matter", "---\ntitle: X\n---")
#   - ("text", "paragraphe normal")
#   - ("sep", "\n\n")  ← séparateur original conservé (Fix #2)
# ──────────────────────────────────────────────

_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _split_preserving_blocks(text: str) -> list[tuple[str, str]]:
    """
    Découpe le texte en segments typés (type, contenu) via une state machine
    ligne par ligne. Gère correctement :
    - Les code fences avec info-string (```python)
    - Les fences imbriquées (fermeture uniquement si même char et >= même longueur)
    - Les backticks inline dans les tableaux (ne déclenchent pas de fence)
    - Les tableaux Markdown comme blocs atomiques (jamais découpés)
    - Préservation des séparateurs originaux (\n\n, \n)
    """
    segments: list[tuple[str, str]] = []

    # ── Extraire le front matter s'il existe ──
    fm_match = _FRONT_MATTER_RE.match(text)
    if fm_match:
        segments.append(("front_matter", fm_match.group(0)))
        text = text[fm_match.end():]

    # ── State machine ──
    lines = text.split("\n")
    in_fence = False
    in_table = False
    fence_char = ""
    fence_len = 0
    current_block: list[str] = []
    current_type = "text"

    def flush_block():
        """Enregistre le bloc courant s'il n'est pas vide."""
        if current_block:
            content = "\n".join(current_block)
            if current_type == "text":
                # Sous-découper le texte brut par doubles sauts de ligne
                # tout en préservant les séparateurs originaux
                sub_parts = re.split(r"(\n\s*\n)", content)
                for sp in sub_parts:
                    if not sp:
                        continue
                    if re.fullmatch(r"\n\s*\n", sp):
                        segments.append(("sep", sp))
                    else:
                        segments.append(("text", sp))
            else:
                segments.append((current_type, content))
            current_block.clear()

    def flush_with_trailing_sep():
        """
        Flush le bloc courant en séparant les lignes vides de fin.
        Les lignes vides trailing deviennent un segment 'sep' pour
        préserver les séparateurs originaux (ex: \\n\\n avant un tableau).
        """
        # Compter les lignes vides à la fin du bloc
        trailing_empty = 0
        for line in reversed(current_block):
            if line.strip() == "":
                trailing_empty += 1
            else:
                break

        if trailing_empty > 0 and trailing_empty < len(current_block):
            # Séparer : le contenu textuel + le séparateur trailing
            sep_lines = current_block[-trailing_empty:]
            del current_block[-trailing_empty:]
            flush_block()
            # Émettre le séparateur : N lignes vides = N+1 newlines
            # (chaque ligne vide = 1 \n entre les lignes du split,
            #  + 1 \n pour la jonction vers la ligne suivante)
            segments.append(("sep", "\n" * (trailing_empty + 1)))
        else:
            flush_block()

    for line in lines:
        if in_fence:
            # Vérifier si cette ligne ferme la fence actuelle
            m = _FENCE_OPEN_RE.match(line.strip())
            if m:
                marker = m.group(1)
                # Fermeture : même caractère, longueur >=, pas d'info-string
                if (marker[0] == fence_char
                        and len(marker) >= fence_len
                        and not m.group(2).strip()):
                    current_block.append(line)
                    in_fence = False
                    flush_block()
                    current_type = "text"
                    continue
            current_block.append(line)

        elif in_table:
            # Continuer le tableau tant que les lignes sont des lignes |...|
            if _TABLE_LINE_RE.match(line):
                current_block.append(line)
            else:
                # Fin du tableau
                in_table = False
                flush_block()
                current_type = "text"
                # Émettre le \n entre la dernière ligne du tableau et cette ligne
                # Si la ligne est vide, c'est le début d'un séparateur \n\n
                if line.strip() == "":
                    segments.append(("sep", "\n\n"))
                else:
                    segments.append(("sep", "\n"))
                    current_block.append(line)

        else:
            # Vérifier si cette ligne ouvre une fence
            stripped = line.strip()
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                marker = m.group(1)
                leading = line[:len(line) - len(line.lstrip())]
                rest_of_line = line[len(leading):]
                if rest_of_line.strip() == stripped and _FENCE_OPEN_RE.match(rest_of_line.strip()):
                    flush_with_trailing_sep()
                    in_fence = True
                    fence_char = marker[0]
                    fence_len = len(marker)
                    current_type = "code"
                    current_block.append(line)
                    continue

            # Vérifier si cette ligne commence un tableau
            if _TABLE_LINE_RE.match(line) and not in_fence:
                # Regarder si c'est un vrai tableau (header + separator)
                # On commence à collecter et on validera au fur et à mesure
                flush_with_trailing_sep()
                in_table = True
                current_type = "table"
                current_block.append(line)
                continue

            current_block.append(line)

    # Flush le dernier bloc
    flush_block()

    return segments


def split_markdown(text: str, max_tokens: int) -> list[tuple[str, str]]:
    """
    Découpe le document Markdown en chunks respectant la limite de tokens,
    tout en préservant les blocs de code, le front matter, et les séparateurs.

    Retourne une liste de (type, contenu) où type est :
    - "code" : bloc de code (ne pas traduire)
    - "front_matter" : front matter YAML (ne pas traduire)
    - "text" : texte à traduire
    - "sep" : séparateur original (ne pas traduire, conserver tel quel)
    """
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    segments = _split_preserving_blocks(text)

    chunks: list[tuple[str, str]] = []
    current_chunk = ""
    current_tokens = 0
    current_type = "text"

    for seg_type, segment in segments:
        # Les blocs spéciaux (code, front_matter, sep, table) sont émis tels quels
        # Note : "table" est traduit mais jamais découpé (atomique)
        if seg_type in ("code", "front_matter", "sep", "table"):
            if current_chunk:
                chunks.append((current_type, current_chunk))
                current_chunk = ""
                current_tokens = 0
            chunks.append((seg_type, segment))
            current_type = "text"
            continue

        # Segment de texte : regrouper en respectant la limite
        seg_tokens = len(encoding.encode(segment))

        if seg_tokens > max_tokens:
            # Segment trop gros → découper ligne par ligne
            if current_chunk:
                chunks.append((current_type, current_chunk))
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
                        chunks.append(("text", current_chunk))
                    current_chunk = line
                    current_tokens = line_tokens
            current_type = "text"
        elif current_tokens + seg_tokens <= max_tokens:
            current_chunk += segment
            current_tokens += seg_tokens
            current_type = "text"
        else:
            chunks.append(("text", current_chunk))
            current_chunk = segment
            current_tokens = seg_tokens
            current_type = "text"

    if current_chunk:
        chunks.append((current_type, current_chunk))

    return chunks


def is_translatable(chunk_type: str) -> bool:
    """Vérifie si un chunk doit être traduit (texte brut + tableaux)."""
    return chunk_type in ("text", "table")


# ──────────────────────────────────────────────
# Validation qualité des traductions (Fix #5)
# ──────────────────────────────────────────────

def validate_translation(original: str, translated: str) -> list[str]:
    """
    Retourne une liste de warnings (vide si OK).
    Vérifie :
      - Réponse vide ou quasi-vide
      - Réponse qui semble tronquée (fin abrupte)
      - Ratio de longueur suspect
    """
    warnings: list[str] = []
    orig_len = len(original.strip())
    trans_len = len(translated.strip())

    if trans_len == 0:
        warnings.append("⚠️ Réponse vide — le modèle n'a produit aucun texte.")
        return warnings

    if orig_len > 20 and trans_len < 5:
        warnings.append("⚠️ Réponse quasi-vide — la traduction semble incomplète.")

    # Ratio de longueur suspect (< 20% ou > 500%)
    if orig_len > 0:
        ratio = trans_len / orig_len
        if ratio < 0.2:
            warnings.append(
                f"⚠️ Ratio de longueur très bas ({ratio:.0%}) — "
                f"traduction potentiellement tronquée."
            )
        elif ratio > 5.0:
            warnings.append(
                f"⚠️ Ratio de longueur très élevé ({ratio:.0%}) — "
                f"le modèle a peut-être halluciné du contenu."
            )

    # Fin abrupte : pas de ponctuation finale alors que l'original en a
    final_punct = set(".!?。！？…」』\"')")
    orig_ends_with_punct = original.strip()[-1] in final_punct if original.strip() else False
    trans_ends_with_punct = translated.strip()[-1] in final_punct if translated.strip() else False
    if orig_ends_with_punct and not trans_ends_with_punct and trans_len > 20:
        warnings.append(
            "⚠️ La traduction ne se termine pas par une ponctuation — "
            "possible troncature (max_new_tokens atteint ?)."
        )

    return warnings


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

    # ── Fix #7 : Glossaire terminologique ──
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

    # ── Fix #4 : Avertissement tokenizer approximatif ──
    st.caption(
        "⚠️ Estimation via **cl100k_base** (GPT-4). Le tokenizer HY-MT peut "
        "donner un comptage différent de ±15 %. La taille réelle des chunks "
        "envoyés au modèle peut varier."
    )

    st.divider()

    # ── Fix #3 : Clés de session state pour résultat + cache ──
    state_key = f"result_{uploaded_file.name}_{uploaded_file.size}_{src_lang}_{tgt_lang}"
    cache_key = f"cache_{uploaded_file.name}_{uploaded_file.size}_{src_lang}_{tgt_lang}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    cache: dict[int, str] = st.session_state[cache_key]

    # Boutons d'action
    btn_col1, btn_col2 = st.columns([3, 1])
    launch = btn_col1.button(
        "🚀 Lancer la traduction", type="primary", use_container_width=True,
    )
    if cache:
        if btn_col2.button("🗑️ Vider le cache", use_container_width=True):
            st.session_state[cache_key] = {}
            st.session_state[state_key] = None
            cache = {}
            st.rerun()

    if launch:
        start_time = time.time()

        # ── Fix #7 : Initialiser le glossaire ──
        glossary_key = f"glossary_{uploaded_file.name}_{uploaded_file.size}_{src_lang}_{tgt_lang}"
        if glossary_key not in st.session_state:
            st.session_state[glossary_key] = {}
        auto_glossary: dict[str, str] = st.session_state[glossary_key]

        # Parser le glossaire manuel
        manual_glossary = {}
        if manual_glossary_text.strip():
            manual_glossary = parse_manual_glossary(manual_glossary_text)

        # Fusionner : le manuel a priorité sur l'automatique
        combined_glossary = {**auto_glossary, **manual_glossary}

        # Découpage
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

        # Traduction
        progress_bar = st.progress(0, text="Démarrage...")
        translated_parts: list[tuple[str, str]] = []
        error_occurred = False
        quality_warnings: list[str] = []

        for i, (chunk_type, chunk_text) in enumerate(chunks):
            progress_bar.progress(
                i / total_chunks,
                text=f"Bloc {i + 1}/{total_chunks}...",
            )

            # Blocs non-traduisibles (code, front_matter, sep)
            if not is_translatable(chunk_type):
                translated_parts.append((chunk_type, chunk_text))
                continue

            # ── Fix #3 : vérifier le cache ──
            if i in cache:
                translated_parts.append(("text", cache[i]))
                continue

            # Préparer le glossaire pour ce chunk (actif si glossaire_enabled)
            chunk_glossary = combined_glossary if glossary_enabled else manual_glossary

            result = translate_with_retry(
                chunk_text, backend_url, selected_model,
                src_lang, tgt_lang,
                temperature, top_k, top_p, repetition_penalty,
                max_response_tokens, request_timeout,
                backend_name,
                glossary=chunk_glossary if chunk_glossary else None,
            )
            if result is None:
                error_occurred = True
                st.error(
                    f"❌ Erreur au bloc {i + 1}/{total_chunks}. "
                    f"**{len(cache)} blocs en cache** — recliquez pour reprendre."
                )
                break

            # Stocker immédiatement en cache
            cache[i] = result

            # ── Fix #7 : Mettre à jour le glossaire automatique ──
            if glossary_enabled:
                align_glossary_from_chunks(chunk_text, result, auto_glossary)
                # Mettre à jour le glossaire combiné pour les blocs suivants
                combined_glossary = {**auto_glossary, **manual_glossary}

            # ── Fix #5 : valider la qualité ──
            chunk_warnings = validate_translation(chunk_text, result)
            for w in chunk_warnings:
                quality_warnings.append(f"Bloc {i + 1} : {w}")

            translated_parts.append(("text", result))

        if not error_occurred:
            progress_bar.progress(1.0, text="Terminé !")
            elapsed = time.time() - start_time

            # ── Fix #2 : jointure fidèle avec séparateurs originaux ──
            result_text = "".join(content for _, content in translated_parts)

            st.session_state[state_key] = result_text
            st.success(
                f"Traduction terminée en **{elapsed:.1f}s** "
                f"({translatable} blocs traduits, "
                f"~{len(encoding.encode(result_text)):,} tokens)"
            )

            # Afficher les warnings qualité s'il y en a
            if quality_warnings:
                with st.expander(
                    f"⚠️ {len(quality_warnings)} avertissement(s) qualité",
                    expanded=False,
                ):
                    for w in quality_warnings:
                        st.warning(w)

    # Affichage du résultat
    result_text = st.session_state.get(state_key)
    if result_text:
        st.divider()

        tab_preview, tab_raw = st.tabs(["📖 Aperçu Markdown", "📝 Texte brut"])
        with tab_preview:
            st.markdown(result_text)
        with tab_raw:
            st.text_area(
                "Résultat brut",
                value=result_text,
                height=400,
                label_visibility="collapsed",
                key=f"raw_{state_key}",
            )
            # Bouton copier via JavaScript (le bouton natif de st.code
            # est cassé dans Streamlit ≥ 1.30 : rendu hors écran)
            import streamlit.components.v1 as components
            _escaped = result_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            components.html(
                f"""
                <button onclick="copyToClipboard()" id="copyBtn"
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
                        const btn = document.getElementById('copyBtn');
                        btn.textContent = '✅ Copié !';
                        setTimeout(() => {{ btn.textContent = '📋 Copier le résultat brut'; }}, 2000);
                    }}).catch(() => {{
                        // Fallback pour les contextes non-sécurisés (HTTP)
                        const ta = document.createElement('textarea');
                        ta.value = text;
                        ta.style.position = 'fixed';
                        ta.style.left = '-9999px';
                        document.body.appendChild(ta);
                        ta.select();
                        document.execCommand('copy');
                        document.body.removeChild(ta);
                        const btn = document.getElementById('copyBtn');
                        btn.textContent = '✅ Copié !';
                        setTimeout(() => {{ btn.textContent = '📋 Copier le résultat brut'; }}, 2000);
                    }});
                }}
                </script>
                """,
                height=50,
            )

        # ── Téléchargement : BytesIO stable pour éviter les .crdownload ──
        import io
        _dl_bytes = result_text.encode("utf-8")
        _dl_buffer = io.BytesIO(_dl_bytes)
        _dl_buffer.name = f"traduit_{uploaded_file.name}"
        _dl_filename = f"traduit_{uploaded_file.name}"
        st.download_button(
            label="📥 Télécharger le fichier traduit (.md)",
            data=_dl_buffer,
            file_name=_dl_filename,
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_{state_key}",
        )

elif not uploaded_file:
    st.info("👆 Chargez un fichier `.md` pour commencer.")
elif src_lang == tgt_lang:
    pass  # Message d'erreur déjà affiché dans la sidebar
elif not selected_model:
    st.warning("⚠️ Sélectionnez ou saisissez un modèle dans la barre latérale.")