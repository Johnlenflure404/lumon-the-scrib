"""
Glossary management — proper noun extraction and terminology alignment.

Extracted verbatim from traduction_app.py (Fix #7).
"""

import re

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
