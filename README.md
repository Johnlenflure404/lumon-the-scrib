# 📄 Lumon The Scrib v2 — Multimodal Translation Workstation

Traducteur local de gros documents, d'images et de flux caméra, propulsé par les modèles [HY-MT (Hunyuan Translation)](https://github.com/Tencent-Hunyuan/HY-MT) et **GLM-OCR**.

**Traduisez des documents de plusieurs milliers de pages, des documents scannés ou même des livres physiques via votre webcam, 100% en local.** Vos données ne quittent jamais votre machine.

---

## ✨ Fonctionnalités (v2)

### 🚀 Nouveautés de la v2 (Multimodalité & Performance)
- **Ingestion Universelle de Documents** — Support natif des fichiers `.md`, `.txt`, `.pdf` (avec extraction native ou OCR de secours), `.docx` et images (`.png`, `.jpg`).
- **OCR Intégré (GLM-OCR)** — Outil de reconnaissance de texte, de tableaux et de formules mathématiques à partir d'images ou de documents scannés.
- **Caméra en Direct** — Capturez des pages de livres ou de documents via votre webcam, extrayez le texte via OCR et traduisez-le instantanément.
- **Exports Multi-Formats** — Téléchargez vos traductions en `.md`, `.txt`, `.pdf`, `.html`, ou `.docx`.
- **Architecture Modulaire** — Codebase refondue pour une meilleure maintenabilité (package `lumon/`).
- **Optimisations de Vitesse** (Architecture prête) — Pooling de connexions HTTP (exécution 2x plus rapide), max_tokens adaptatif, backoff réduit et support de traduction de blocs en parallèle.

### 🛡️ Moteur de Traduction HY-MT (Hérité et Préservé)
- **33 langues** supportées (anglais, français, chinois, japonais, arabe, etc.).
- **Découpage intelligent (State machine)** — Sépare le texte, préserve les blocs de code et le front matter YAML intacts.
- **Tableaux atomiques** — Les tableaux Markdown ne sont jamais découpés au milieu.
- **Glossaire terminologique** — Cohérence automatique des noms propres via le template d'intervention HY-MT.
- **Validation qualité** — Détection des réponses vides, tronquées ou aux hallucinations probables.
- **Multi-backend** — Compatible **LM Studio** et **Ollama** avec un système unifié de sélection.

---

## 📋 Prérequis

| Composant | Description |
|---|---|
| **Python** | 3.10 ou supérieur |
| **Backend LLM** | LM Studio ou Ollama (au choix) |
| **Modèle de Traduction** | HY-MT1.5-1.8B ou HY-MT1.5-7B (GGUF ou Ollama) |
| **Modèle OCR (Vision)** | GLM-OCR (requis pour traduire des images, PDF scannés et caméra) |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd lumon-the-scrib
```

### 2. Installer les backends et modèles

Vous pouvez utiliser Ollama, LM Studio, ou les deux.

#### Option A : LM Studio (Traduction & OCR via vision API)
1. Téléchargez LM Studio sur [lmstudio.ai](https://lmstudio.ai).
2. Cherchez et téléchargez `HY-MT1.5` (ex: `HY-MT1.5-1.8B-GGUF`).
3. Cherchez et téléchargez `GLM-OCR` (ou un modèle Vision performant équivalent).
4. Chargez les modèles en mémoire et démarrez le serveur local (onglet **Developer** `<>`). Le serveur tourne généralement sur `http://localhost:1234`.

#### Option B : Ollama (Traduction & OCR natif)
1. Installez Ollama depuis [ollama.com](https://ollama.com).
2. Téléchargez les modèles nécessaires :
```bash
ollama pull hf.co/tencent/HY-MT1.5-1.8B-GGUF
ollama pull glm-ocr:latest
```

---

## ▶️ Lancement

Une seule commande gère l'installation des dépendances (PyMuPDF, python-docx, Pillow, fpdf2, markdown...) et lance l'app :

```bash
./run.sh
```

L'application s'ouvrira dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📖 Utilisation détaillée

L'interface est découpée en **4 onglets principaux** :

### 1. 📄 Texte / Markdown
Le mode original de Lumon The Scrib. Collez du texte brut ou importez un fichier `.md`/`.txt`. Idéal pour les traductions ultra-rapides et fidèles au format.

### 2. 📑 Documents (PDF, DOCX, Images)
Importez vos documents complexes.
- Les **DOCX** verront leurs titres et tableaux extraits.
- Les **PDF textuels** sont extraits directement.
- Les **PDF scannés** ou les **Images** font appel à **GLM-OCR**. Vous pouvez choisir le prompt OCR désiré dans la barre latérale (Texte régulier, Extraction de tableaux, Préservation de requêtes mathématiques).

### 3. 📷 Caméra
Utilisez la webcam de votre ordinateur pour traduire en direct.
- Prenez une photo d'une page physique.
- GLM-OCR en extrait le texte.
- HY-MT traduit le tout, tout en lissant les erreurs typographiques de l'OCR.

### 4. ℹ️ À Propos
Pour consulter les aides en ligne, la documentation de HY-MT et voir les optimisations système.

### 🔧 Rendu et Exports
Une fois la traduction achevée, l'interface affiche une progression détaillée, et vous permet d'exporter la traduction en un clic vers :
- **Markdown (.md)**
- **Texte brut (.txt)**
- **PDF (.pdf)** (Généré proprement avec support Unicode et ajustement des marges/tableaux)
- **HTML (.html)**
- **Word (.docx)**

---

## ⚙️ Configuration & Performances (Sidebar)

### Optimisations de Performances
La sidebar intègre des paramètres de configuration et d'optimisations vitaux :
- **Choix du Backend et des URL** : Permet de router l'OCR vers Ollama et la Traduction vers LM Studio (ou vice-versa).
- **Tokens max par bloc** : Ajustable selon la taille de votre contexte.
- **Réglages du glossaire** : Activer/Désactiver l'extraction automatique des entités nommées.

**Astuces pour accélérer (Tips) :**
- Utilisez une connexion HTTP persistante avec Ollama (`OLLAMA_KEEP_ALIVE=-1`).
- Activez **Speculative Decoding** dans LM Studio si disponible (accélère l'inférence de X2).
- Sur Ollama, lancez `OLLAMA_NUM_PARALLEL=4` pour permettre de multiples requêtes de traduction concurrentes dans le futur mode parallèle.

---

## 📁 Structure du projet (Modulaire)

La V2 introduit un code propre et factorisé (`lumon/`) par rapport au script monolithique de la V1 :

```
lumon-the-scrib/
├── app.py                    # Application Streamlit principale (4 onglets multi-modaux)
├── run.sh                    # Lanceur auto-installateur
├── requirements.txt          # Nouvelles dépendances
├── lumon/
│   ├── __init__.py
│   ├── config.py             # Constantes et sélection de backend
│   ├── translation.py        # Moteur HY-MT avec streaming optimisé
│   ├── chunking.py           # État de découpe Markdown précis
│   ├── glossary.py           # Logique d'alignement terminologique
│   ├── validation.py         # Outils de vérification de qualité
│   ├── ocr.py                # Wrapper client GLM-OCR (API OpenAI et Ollama)
│   ├── document.py           # Parsing PDF, DOCX, parsing d'images vers Markdown
│   └── export.py             # Moteurs de rendu PDF, DOCX, HTML
└── traduction_app.py         # (Legacy) Version V1 préservée à titre d'archive
```

---

## ❓ Dépannage

| Problème | Solution |
|---|---|
| **L'OCR est indisponible** | Vérifiez que GLM-OCR est bien chargé dans Ollama ou LM Studio et que son port correspond dans la barre latérale. |
| **Crash export PDF sur tableau** | C'est corrigé ! L'export PDF nettoie intelligemment les sauts de lignes des tableaux et redimensionne la largeur effective (`epw`). |
| **Traduction coupée en plein milieu** | Changez les paramètres avancés de HY-MT (`max_tokens` > 3000) ou baissez les tokens par bloc. Le cache permet de reprendre la traduction à l'endroit bloqué en recliquant sur "Lancer". |

---

## 📜 Licence

Lumon The Scrib utilise le framework Streamlit. La logique de validation et de traduction emploie les recommandations du modèle [HY-MT](https://github.com/Tencent-Hunyuan/HY-MT) de Tencent. Documentation du modèle incluse pour référence.
