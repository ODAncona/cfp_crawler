# 📢 CFP - Extraction et Analyse Automatique des Call For Papers

## 📌 Description

**cfp** est un projet permettant d'extraire et d'analyser automatiquement les appels à communications (CFP) depuis [WikiCFP](http://www.wikicfp.com/) afin d'identifier les conférences les plus pertinentes en fonction d'un résumé scientifique donné. Le projet s'appuie sur un modèle de langage (LLM) pour évaluer la pertinence des conférences.

## 📂 Arborescence du Projet

```
cfp
├── src
│   └── cfp
│       ├── __init__.py
│       ├── main.py
│       └── prompt.md
├── pyproject.toml
└── README.md
```

## ✨ Fonctionnalités

- 🔎 **Recherche automatique** de conférences sur WikiCFP en fonction d'un mot-clé.
- 📑 **Extraction des détails** des conférences : Titre, Acronyme, Date, Lieu, Date limite, Description.
- 🧠 **Évaluation de la pertinence** avec un LLM en comparant les détails des conférences avec le résumé fourni.
- 💾 **Sauvegarde progressive** des conférences pertinentes dans un fichier CSV.
- 📊 **Suivi de l'exécution** avec `tqdm` et `colorlog`.

## 🚀 Installation

### 📥 Prérequis

1. Assurez-vous d'avoir **Python 3.8+** installé.
2. Clonez le projet :

   ```bash
   git clone https://github.com/votre-repo/cfp.git
   cd cfp
   ```

3. Installez les dépendances avec **Rye** :

   ```bash
   rye sync
   ```

4. Configurez votre clé API OpenAI dans un fichier `.env` :

   ```bash
   echo "OPENAI_API_KEY=your_api_key" > .env
   ```

## 🛠 Utilisation

Exécutez le script avec :

```bash
python src/cfp/main.py chemin/vers/abstract.txt mot-clé -o fichier_sortie.csv
```

### 📌 Exemple

```bash
python src/cfp/main.py abstract.txt machine-learning -o conferences_ml.csv
```

## 📊 Structure du CSV généré

| Title | Acronym | When | Where | Deadline | Score | Justification |
|-------|---------|------|-------|----------|-------|--------------|
| Conférence A | CONF-A | 12-14 Juin 2025 | Paris, France | 1 Mars 2025 | 8 | Pertinent pour votre domaine |
| Conférence B | CONF-B | 5-7 Octobre 2025 | New York, USA | 15 Avril 2025 | 6 | Sujet légèrement éloigné mais intéressant |

## 🔍 Développement

Si vous souhaitez contribuer ou modifier le projet :

```bash
rye run python -m src.cfp.main abstract.txt ai -o results.csv
```

## 📝 Licence

Ce projet est sous licence GPL - voir le fichier `LICENSE` pour plus de détails.

---

🚀 **Développé par Olivier D'Ancona**
