"""
HY-MT prompt building and streaming translation engine.

Extracted verbatim from traduction_app.py.
Prompt templates, inference parameters, and API interaction logic
are preserved exactly as documented by HY-MT.
"""

import json
import time
import requests

from .config import (
    ZH_TARGET_NAMES,
    EN_TARGET_NAMES,
    MAX_RETRIES,
)


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
        target_name = ZH_TARGET_NAMES.get(tgt_lang, tgt_lang)
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
        target_name = EN_TARGET_NAMES.get(tgt_lang, tgt_lang)
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
                raise RuntimeError(
                    f"Échec après {max_retries} tentatives : {e}"
                ) from e
    return None
