# 📄 Lumon The Scrib

Traducteur local de gros documents Markdown, propulsé par le modèle [HY-MT (Hunyuan Translation)](https://github.com/Tencent-Hunyuan/HY-MT) de Tencent.

**Traduisez des documents Markdown de plusieurs milliers de pages en local**, sans envoyer vos données sur Internet.

---

## ✨ Fonctionnalités

- **33 langues** supportées (anglais, français, chinois, japonais, arabe, etc.)
- **Optimisé pour les très gros documents** — Découpage intelligent par state machine, streaming, retry automatique
- **100 % local** — Vos données ne quittent jamais votre machine
- **Préservation du Markdown** — Blocs de code, tableaux, front matter YAML conservés intacts
- **Tableaux atomiques** — Les tableaux Markdown ne sont jamais découpés au milieu
- **Glossaire terminologique** — Cohérence automatique des noms propres entre les blocs (template HY-MT officiel)
- **Cache de traduction** — Reprise automatique après échec, sans re-traduire les blocs déjà traités
- **Validation qualité** — Détection des réponses vides, tronquées ou au ratio suspect
- **Multi-backend** — Compatible LM Studio et Ollama (payload adapté nativement)
- **Encodage UTF-8 robuste** — Décodage forcé pour éviter les artefacts sur tous les caractères
- **Interface simple** — Upload, clic, téléchargement

---

## 📋 Prérequis

| Composant | Version minimale |
|---|---|
| **Python** | 3.10+ |
| **LM Studio** ou **Ollama** | Dernière version |
| **Modèle HY-MT** | HY-MT1.5-1.8B ou HY-MT1.5-7B |

### Choix du modèle

| Modèle | RAM requise | Idéal pour |
|---|---|---|
| **HY-MT1.5-1.8B-GGUF** | ~2 Go | Machines modestes, traduction rapide |
| **HY-MT1.5-1.8B** | ~4 Go | Bon compromis vitesse/qualité |
| **HY-MT1.5-7B-GGUF** | ~6 Go | Meilleure qualité |
| **HY-MT1.5-7B** | ~14 Go | Qualité maximale (GPU recommandé) |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd lumon-the-scrib
```

### 2. Installer un backend LLM

Vous avez le choix entre **LM Studio** ou **Ollama**. Installez l'un des deux.

#### Option A : LM Studio (recommandé pour débutants)

1. **Télécharger** LM Studio depuis [lmstudio.ai](https://lmstudio.ai)
2. **Installer** et lancer l'application
3. **Chercher le modèle** : dans la barre de recherche, tapez `HY-MT1.5`
4. **Télécharger** le modèle souhaité (ex: `HY-MT1.5-1.8B-GGUF`)
5. **Charger le modèle** : cliquez dessus pour le charger en mémoire
6. **Démarrer le serveur** :
   - Allez dans l'onglet **"Developer"** (icône `<>`)
   - Cliquez sur **"Start Server"**
   - Le serveur démarre sur `http://localhost:1234`

> 💡 Le serveur LM Studio doit rester ouvert pendant toute l'utilisation de l'app.

#### Option B : Ollama

1. **Installer** Ollama depuis [ollama.com](https://ollama.com)
2. **Télécharger le modèle** :

```bash
ollama pull hf.co/tencent/HY-MT1.5-1.8B-GGUF
```

3. **Vérifier** que le modèle est bien installé :

```bash
ollama list
```

Le serveur Ollama écoute par défaut sur `http://localhost:11434`.

> 📝 L'application adapte automatiquement le format de requête pour Ollama (les paramètres `top_k` et `repeat_penalty` sont transmis dans l'objet `options` conformément à l'API Ollama).

---

## ▶️ Lancement

Une seule commande :

```bash
./run.sh
```

Le script vérifie automatiquement les dépendances Python, les installe si nécessaire, et lance l'application.

> 💡 **Alternative manuelle** :
> ```bash
> pip3 install -r requirements.txt
> python3 -m streamlit run traduction_app.py
> ```

L'application s'ouvre dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📖 Utilisation

### 1. Configurer le backend (sidebar gauche)

- **Backend LLM** : Sélectionnez `LM Studio` ou `Ollama`
- **URL du serveur** : Laissez la valeur par défaut sauf port personnalisé
- **Modèle** : Détecté automatiquement si le backend tourne. Sinon, saisie manuelle.

### 2. Choisir les langues

- **Langue source** : La langue du document original
- **Langue cible** : La langue de destination

### 3. Configurer le glossaire (optionnel)

Dans la sidebar, section **"📖 Glossaire"** :

- **Glossaire automatique** (activé par défaut) — L'app détecte les noms propres dans chaque bloc traduit et les injecte dans les prompts suivants via le [template terminologique HY-MT](HY-MT_MODEL_DOC.md). Cela assure la cohérence des noms propres entre les blocs (ex: un personnage, une entreprise).

- **Glossaire personnalisé** — Saisissez des termes manuellement, un par ligne :
  ```
  Tokyo → 東京
  OpenAI → OpenAI
  Machine Learning -> Apprentissage automatique
  ```
  Le glossaire manuel a **priorité** sur l'automatique.

> 📝 L'extraction automatique fonctionne bien pour les langues latines (FR, EN, ES, DE…). Pour les langues sans majuscules (ZH, JA, KO), utilisez le glossaire manuel.

### 4. Charger et traduire

1. Cliquez sur **"Browse files"** ou glissez-déposez votre fichier `.md`
2. Cliquez sur **🚀 Lancer la traduction**
3. La barre de progression montre l'avancement bloc par bloc

Les éléments suivants sont automatiquement préservés sans traduction :
- **Blocs de code** (``` et ~~~, y compris imbriqués)
- **Front matter YAML** (`---`)

Les **tableaux Markdown** sont traduits en bloc complet (jamais découpés au milieu).

### 5. Reprise après erreur

Si la traduction échoue au bloc N, les blocs déjà traduits sont **mis en cache automatiquement**. Cliquez à nouveau sur **🚀 Lancer la traduction** pour reprendre là où vous vous êtes arrêté.

- Le message d'erreur indique le nombre de blocs en cache
- Le bouton **🗑️ Vider le cache** permet de forcer une re-traduction complète

### 6. Vérifier la qualité

Après la traduction, l'app affiche automatiquement les **avertissements qualité** (si applicable) :

| Avertissement | Signification |
|---|---|
| ⚠️ Réponse vide | Le modèle n'a produit aucun texte |
| ⚠️ Ratio très bas (< 20 %) | Traduction potentiellement tronquée |
| ⚠️ Ratio très élevé (> 500 %) | Hallucination probable |
| ⚠️ Pas de ponctuation finale | Possible troncature (`max_new_tokens` atteint) |

### 7. Récupérer le résultat

- **Aperçu Markdown** — Visualisez le rendu final
- **Texte brut** — Voyez le code Markdown source
- **📥 Télécharger** — Sauvegardez le fichier traduit

---

## ⚙️ Paramètres avancés

Accessibles dans la sidebar, sous **"🔧 Paramètres avancés"** :

| Paramètre | Défaut | Description |
|---|---|---|
| **Tokens max par bloc** | 1500 | Taille max de chaque chunk envoyé au modèle |
| **Température** | 0.7 | Créativité. Plus bas = plus littéral |
| **Top-K** | 20 | Nombre de tokens candidats à chaque étape |
| **Top-P** | 0.6 | Seuil de probabilité cumulative |
| **Repetition penalty** | 1.05 | Pénalise les répétitions |
| **Tokens max par réponse** | 2048 | Longueur max de la réponse par bloc |
| **Timeout** | 120s | Temps max d'attente par bloc |

> Les valeurs par défaut sont celles **recommandées par la documentation officielle HY-MT**.

> ⚠️ Le comptage de tokens affiché utilise `cl100k_base` (GPT-4). Le tokenizer HY-MT peut donner un résultat différent de ±15 %.

---

## 🌍 Langues supportées (33)

| Langue | Code | Langue | Code |
|---|---|---|---|
| Chinese | `zh` | Polish | `pl` |
| English | `en` | Czech | `cs` |
| French | `fr` | Dutch | `nl` |
| Portuguese | `pt` | Khmer | `km` |
| Spanish | `es` | Burmese | `my` |
| Japanese | `ja` | Persian | `fa` |
| Turkish | `tr` | Gujarati | `gu` |
| Russian | `ru` | Urdu | `ur` |
| Arabic | `ar` | Telugu | `te` |
| Korean | `ko` | Marathi | `mr` |
| Thai | `th` | Hebrew | `he` |
| Italian | `it` | Bengali | `bn` |
| German | `de` | Tamil | `ta` |
| Vietnamese | `vi` | Ukrainian | `uk` |
| Malay | `ms` | Tibetan | `bo` |
| Indonesian | `id` | Kazakh | `kk` |
| Filipino | `tl` | Mongolian | `mn` |
| Hindi | `hi` | Uyghur | `ug` |
| Traditional Chinese | `zh-Hant` | Cantonese | `yue` |

---

## 🏗️ Architecture technique

### Découpage intelligent (state machine)

Le document est découpé par une **machine à états ligne par ligne** :

1. **Front matter** détecté et isolé (non traduit)
2. **Code fences** détectées par type et longueur (gestion des fences imbriquées ``````)
3. **Tableaux** détectés comme blocs atomiques (lignes `|...|` consécutives, jamais découpés)
4. **Texte** sous-découpé par paragraphes, avec **séparateurs originaux préservés**
5. Les chunks sont regroupés en respectant la limite de tokens

### Glossaire terminologique

L'application utilise le **template d'intervention terminologique** de la doc officielle HY-MT :

```
参考下面的翻译：
{source_term} 翻译成 {target_term}

将以下文本翻译为{target_language}...
```

Les noms propres sont extraits par regex heuristique (majuscules latines, mixed-case, all-caps) et le glossaire est filtré pour chaque chunk : seuls les termes présents dans le texte source du chunk sont injectés dans le prompt.

### Payload multi-backend

| Paramètre | LM Studio | Ollama |
|---|---|---|
| `top_k` | racine du payload | `options.top_k` |
| `repetition_penalty` | racine du payload | `options.repeat_penalty` |

---

## 📁 Structure du projet

```
lumon-the-scrib/
├── run.sh               # Lanceur (une commande pour tout démarrer)
├── traduction_app.py    # Application principale (~1000 lignes)
├── requirements.txt     # Dépendances Python (streamlit, requests, tiktoken)
├── README.md            # Ce fichier
└── HY-MT_MODEL_DOC.md   # Documentation officielle du modèle HY-MT
```

---

## ❓ Dépannage

| Problème | Solution |
|---|---|
| **"Aucun modèle détecté"** | Vérifiez que votre backend (LM Studio / Ollama) est lancé et qu'un modèle est chargé |
| **Traduction très lente** | Augmentez le Timeout. Utilisez le modèle `1.8B` au lieu du `7B` |
| **Le modèle coupe ses réponses** | Réduisez *Tokens max par bloc* (ex: 800). Augmentez *Tokens max par réponse* (ex: 4096) |
| **Caractères corrompus (脙漏, 芒聙聶)** | Normalement corrigé. Vérifiez les logs `[UTF-8 DEBUG]` dans le terminal |
| **Traduction interrompue** | Les blocs déjà traduits sont en cache — recliquez sur 🚀 pour reprendre |
| **Noms propres incohérents** | Ajoutez les termes dans le glossaire personnalisé (sidebar → 📖 Glossaire) |
| **Tableau cassé** | Les tableaux sont désormais traités comme blocs atomiques : non découpés |
| **`streamlit` introuvable** | Utilisez `python3 -m streamlit run traduction_app.py` ou `./run.sh` |
| **L'app ne se lance pas** | Vérifiez Python 3.10+ : `python3 --version`. Réinstallez : `pip3 install -r requirements.txt` |

---

## 📜 Licence

Ce projet utilise le modèle HY-MT de Tencent. Consultez la [documentation du modèle](HY-MT_MODEL_DOC.md) pour les conditions d'utilisation.
