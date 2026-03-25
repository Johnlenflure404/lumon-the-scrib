"""
Translation quality validation.

Extracted verbatim from traduction_app.py (Fix #5).
"""


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
