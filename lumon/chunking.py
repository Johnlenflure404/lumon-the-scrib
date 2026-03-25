"""
Intelligent Markdown chunking via state machine.

Extracted verbatim from traduction_app.py (Fix #1 + #2).
Handles code fences, front matter, tables, and paragraph splitting
while preserving original separators.
"""

import re
import tiktoken

from .config import TIKTOKEN_ENCODING

# ──────────────────────────────────────────────
# Regex patterns
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
