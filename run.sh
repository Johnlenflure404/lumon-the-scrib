#!/bin/bash
# ──────────────────────────────────────────────
# Lanceur HY-MT Traducteur Markdown
# Usage : ./run.sh
# ──────────────────────────────────────────────

set -e

# Répertoire du script (fonctionne même si lancé depuis un autre dossier)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "📄 Lumon The Scrib"
echo "────────────────────────────"

# Vérifier que Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé."
    echo "   Installez-le depuis https://www.python.org/downloads/"
    exit 1
fi

# Installer les dépendances si nécessaire (silencieux si déjà installées)
echo "📦 Vérification des dépendances..."
pip3 install -q -r "$SCRIPT_DIR/requirements.txt"

# Lancer Streamlit
echo "🚀 Lancement de l'application..."
echo ""
python3 -m streamlit run "$SCRIPT_DIR/traduction_app.py"
