#!/usr/bin/env python
"""
Script d'extraction et d'analyse automatique des CFP (Call For Papers)
pour trouver des conférences pertinentes en fonction d'un résumé scientifique.

Fonctionnalités :
- Recherche de conférences sur WikiCFP en fonction d'un mot-clé sur l'ensemble des pages de résultats.
- Extraction des détails de chaque conférence (Titre, Acronyme, Quand, Où, Date limite, Description).
- Évaluation de la pertinence de la conférence à l'aide d'un LLM (ChatOpenAI)
  en comparant le résumé fourni avec les détails de la conférence.
- Sauvegarde progressive des conférences pertinentes (score > 5) dans un fichier CSV.
- Suivi de l'exécution via un logger coloré et des barres de progression imbriquées (tqdm).
  
Utilisation :
    python find_conferences.py chemin/vers/abstract.txt computing -o conferences.csv
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import colorlog
import re
from pydantic import BaseModel, Field
from tqdm import tqdm

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ~~~ Modèle de Données ~~~

class CFP(BaseModel):
    """Modèle pour stocker les informations d'un CFP récupéré depuis WikiCFP."""
    title: str = Field(..., description="Titre de la conférence")
    acronym: str = Field(..., description="Acronyme de la conférence")
    when: str = Field(..., description="Date de la conférence")
    where: str = Field(..., description="Lieu de la conférence")
    deadline: str = Field(..., description="Date limite de soumission")
    description: str = Field(..., description="Description de la conférence")

class CFPMatchResult(BaseModel):
    """
    Modèle structuré pour la réponse du LLM.
    Le LLM doit renvoyer un objet JSON avec :
        - score : un entier entre 0 et 10
        - justification : un texte court justifiant l'évaluation
    """
    score: int = Field(..., description="Score de pertinence (0-10)")
    justification: str = Field(..., description="Justification de l'évaluation")

# ~~~ Chaîne d'Évaluation ~~~

prompt_evaluate = PromptTemplate(
    input_variables=["abstract", "title", "acronym", "when", "where", "deadline"],
    template="""
Vous êtes un expert en évaluation de la pertinence des conférences pour un article scientifique.

Voici le résumé de notre travail :
{abstract}

Et voici les détails de la conférence :
- Titre : {title}
- Acronyme : {acronym}
- Quand : {when}
- Où : {where}
- Date limite : {deadline}

Veuillez évaluer la pertinence de cette conférence pour notre travail sur une échelle de 0 à 10.
Fournissez uniquement un objet JSON avec les clés "score" (un entier entre 0 et 10) et "justification" (un court texte expliquant votre évaluation).
"""
)

# Initialisation du LLM avec l'API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# Chaîne d'évaluation qui combine le prompt et le LLM pour obtenir une sortie structurée
chain_evaluate = prompt_evaluate | llm.with_structured_output(CFPMatchResult)

# ~~~ Fonctions d'Extraction ~~~

def get_total_pages(keyword: str) -> int:
    """
    Récupère le nombre total de pages de résultats pour le mot-clé donné.
    Extrait le nombre de pages depuis le texte "Total of ... CFPs in X pages".
    """
    search_url = f"http://www.wikicfp.com/cfp/call?conference={keyword}"
    logger.info(f"Extraction du nombre total de pages pour le mot-clé : {keyword}")
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Parcours des td centrés pour trouver le texte contenant "Total of"
    for td in soup.find_all("td", align="center"):
        text = td.get_text(strip=True)
        if "Total of" in text:
            m = re.search(r"Total of\s+\d+\s+CFPs in\s+(\d+)\s+pages", text)
            if m:
                total_pages = int(m.group(1))
                logger.info(f"Nombre total de pages trouvé : {total_pages}")
                return total_pages
    logger.warning("Nombre de pages non trouvé, utilisation de 1 comme valeur par défaut.")
    return 1


def search_wikicfp_page(keyword: str, page: int) -> set:
    """
    Recherche sur WikiCFP pour le mot-clé donné à la page spécifiée.
    Retourne un ensemble d'URLs de pages détail des CFP.
    """
    search_url = f"http://www.wikicfp.com/cfp/call?conference={keyword}&page={page}"
    logger.info(f"Recherche de CFP sur la page {page} pour le mot-clé : {keyword}")
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    base_url = "http://www.wikicfp.com"
    links = soup.find_all("a", href=lambda href: href and "event.showcfp" in href)
    detail_urls = {base_url + link.get("href") for link in links}
    logger.info(f"{len(detail_urls)} CFP trouvés sur la page {page}.")
    return detail_urls

def parse_cfp_detail_page(url: str) -> CFP:
    """
    Récupère et analyse la page détail d'un CFP sur WikiCFP.
    Extraction des informations : titre, acronyme, date, lieu, deadline et description.
    """
    logger.info(f"Ouverture de la page CFP : {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extraction du titre depuis la balise <span property="v:description">
    title_tag = soup.find("span", property="v:description")
    title = title_tag.get_text(strip=True) if title_tag else "N/A"

    # Extraction de l'acronyme depuis la balise <span property="v:summary">
    acronym_tag = soup.find("span", property="v:summary")
    acronym = acronym_tag["content"].strip() if acronym_tag and acronym_tag.has_attr("content") else "N/A"
    
    # Extraction des informations "When", "Where" et "Submission Deadline" depuis la table de classe "gglu"
    when, where, deadline = "N/A", "N/A", "N/A"
    tables = soup.find_all("table", class_="gglu")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            header = row.find("th")
            if header:
                header_text = header.get_text(strip=True).lower()
                cell = row.find("td")
                if cell:
                    cell_text = cell.get_text(strip=True)
                    if "when" in header_text:
                        when = cell_text
                    elif "where" in header_text:
                        where = cell_text
                    elif "submission deadline" in header_text:
                        deadline = cell_text

    # Extraction de la description dans le div de classe "cfp"
    cfp_div = soup.find("div", class_="cfp")
    description = cfp_div.get_text(separator="\n").strip() if cfp_div else "N/A"
    
    logger.debug(f"CFP extrait : Titre = {title}, Acronyme = {acronym}, When = {when}, Where = {where}, Deadline = {deadline}")
    return CFP(
        title=title,
        acronym=acronym,
        when=when,
        where=where,
        deadline=deadline,
        description=description
    )

# ~~~ Fonction Principale ~~~

def main():
    parser = argparse.ArgumentParser(
        description="Recherche des conférences sur WikiCFP et évaluation de leur pertinence par rapport à un résumé."
    )
    parser.add_argument(
        "abstract_file",
        help="Chemin vers le fichier contenant le résumé (abstract) de votre travail."
    )
    parser.add_argument(
        "keyword",
        help="Mot-clé de recherche à utiliser sur WikiCFP (exemple : computing)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="conferences.csv",
        help="Chemin vers le fichier CSV de sortie (par défaut : conferences.csv)."
    )
    args = parser.parse_args()

    # Lecture du résumé depuis le fichier fourni
    try:
        with open(args.abstract_file, "r", encoding="utf-8") as f:
            abstract_text = f.read().strip()
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier abstract : {e}")
        return

    # Récupération du nombre total de pages
    total_pages = get_total_pages(args.keyword)
    evaluated_results = []

    # Boucle sur chaque page de résultats avec une barre de progression externe
    for page in tqdm(range(1, total_pages + 1), desc="Pages", unit="page"):
        detail_urls = search_wikicfp_page(args.keyword, page)
        # Boucle interne pour chaque CFP sur la page, avec une barre de progression interne
        for url in tqdm(detail_urls, desc=f"Page {page} - CFP", leave=False, unit="cfp"):
            try:
                cfp = parse_cfp_detail_page(url)
                logger.info(f"Évaluation du CFP : {cfp.title}")
                evaluation = chain_evaluate.invoke({
                    "abstract": abstract_text,
                    "title": cfp.title,
                    "acronym": cfp.acronym,
                    "when": cfp.when,
                    "where": cfp.where,
                    "deadline": cfp.deadline
                })
                if evaluation.score > 5:
                    evaluated_results.append({
                        "Title": cfp.title,
                        "Acronym": cfp.acronym,
                        "When": cfp.when,
                        "Where": cfp.where,
                        "Deadline": cfp.deadline,
                        "Score": evaluation.score,
                        "Justification": evaluation.justification
                    })
                    logger.info(f"CFP pertinent trouvé : {cfp.title} (Score : {evaluation.score})")
                else:
                    logger.info(f"CFP non pertinent : {cfp.title} (Score : {evaluation.score})")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {url} : {e}")
        # Sauvegarde intermédiaire après chaque page traitée
        if evaluated_results:
            df = pd.DataFrame(evaluated_results)
            df.to_csv(args.output, index=False, encoding="utf-8")
            logger.info(f"Résultats sauvegardés temporairement dans {args.output}")

    if evaluated_results:
        df = pd.DataFrame(evaluated_results)
        df.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"Résultats finaux sauvegardés dans {args.output}")
    else:
        logger.warning("Aucun CFP pertinent n'a été trouvé.")

if __name__ == "__main__":
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    main()
