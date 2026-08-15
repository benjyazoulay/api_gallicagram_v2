import os
import re
import sqlite3
import contextlib
import io
import math
from typing import Optional, List, Dict, Any
from datetime import date, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# Configuration du backend Matplotlib sans interface graphique (obligatoire en environnement serveur)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

app = FastAPI(
    title="Gallicagram API v2",
    description="API Gallicagram - Accès optimisé aux séries temporelles d'usage de mots dans les corpus de Gallica, avec endpoints dédiés pour les graphiques.",
    version="2.2.0",
    docs_url="/v2/docs",            # Déplace le Swagger UI de /docs à /v2/docs
    redoc_url="/v2/redoc",          # Déplace ReDoc de /redoc à /v2/redoc
    openapi_url="/v2/openapi.json", # Déplace le schéma JSON de /openapi.json à /v2/openapi.json
    root_path="/guni"               # Indique à Swagger d'utiliser le préfixe /guni pour charger le schéma JSON
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# CONSTANTE DE DESCRIPTION DES CORPUS
# ---------------------------------------------------------------------------

CORPUS_DESCRIPTION = """Identifiant du corpus à interroger.

| Corpus | Code | Période | N-gram max | Résolutions disponibles |
| :--- | :--- | :--- | :---: | :--- |
| **Le Monde** | `lemonde` | 1944–2023 | 4 | jour, mois, annee |
| **Le Monde par rubriques** | `lemonde_rubriques` | 1944–2023 | 4 | jour, mois, annee |
| **Presse Gallica** | `presse` | 1789–1950 | 3 | mois, annee |
| **Livres Gallica** | `livres` | 1600–1940 | 5 | annee |
| **Persée** | *(via /query_persee)* | 1789–2023 | 2 | annee |
| **Deutsches Zeitungsportal** | `ddb` | 1780–1950 | 2 | mois, annee |
| **American Stories** | `american_stories` | 1798–1963 | 3 | annee |
| **Journal de Paris** | `paris` | 1777–1827 | 2 | jour, mois, annee |
| **Moniteur Universel** | `moniteur` | 1789–1869 | 2 | jour, mois, annee |
| **Journal des Débats** | `journal_des_debats` | 1789–1944 | 1 | jour, mois, annee |
| **La Presse** | `la_presse` | 1836–1869 | 2 | jour, mois, annee |
| **Le Constitutionnel** | `constitutionnel` | 1821–1913 | 2 | jour, mois, annee |
| **Le Figaro** | `figaro` | 1854–1952 | 2 | jour, mois, annee |
| **Le Temps** | `temps` | 1861–1942 | 2 | jour, mois, annee |
| **Le Petit Journal** | `petit_journal` | 1863–1942 | 2 | jour, mois, annee |
| **Le Petit Parisien** | `petit_parisien` | 1876–1944 | 2 | jour, mois, annee |
| **L'Humanité** | `huma` | 1904–1952 | 2 | jour, mois, annee |
| **Subtitles (FR)** | `subtitles` | 1935–2020 | 3 | annee |
| **Subtitles (EN)** | `subtitles_en` | 1930–2020 | 3 | annee |
| **Rap (Genius)** | `rap` | 1989–2024 | 5 | annee |"""


# ---------------------------------------------------------------------------
# MODÈLES PYDANTIC DE RÉPONSE
# ---------------------------------------------------------------------------

class TermFrequency(BaseModel):
    """Modèle unifié pour les fréquences de termes non-temporelles (joker, associés)"""
    gram: str
    tot: int

class ColumnarSeriesGroup(BaseModel):
    """Groupe de données temporelles au format colonnaire (tableaux parallèles)"""
    gram: str
    rubrique: Optional[str] = None
    dates: List[str]
    n: List[int]
    total: List[int]
    freq: List[float]

class OptimizedQueryResponse(BaseModel):
    """Enveloppe standard optimisée pour les requêtes temporelles"""
    corpus: str
    resolution: str
    results: List[ColumnarSeriesGroup]

class PerseeColumnarSeriesGroup(BaseModel):
    """Groupe colonnaire dédié pour Persée"""
    gram: str
    revue: Optional[str] = None
    dates: List[str]
    n: List[int]
    total: List[int]
    freq: List[float]

class PerseeOptimizedQueryResponse(BaseModel):
    """Enveloppe standard optimisée pour Persée"""
    corpus: str = "persee"
    results: List[PerseeColumnarSeriesGroup]

class SourceRapResultItem(BaseModel):
    """Modèle pour l'endpoint de métadonnées de paroles de rap"""
    title: str
    year: int
    artist: str
    url: str
    counts: int
    context_left: str
    pivot: str
    context_right: str


# ---------------------------------------------------------------------------
# UTILITAIRES DE CONNEXION ET TRAITEMENT
# ---------------------------------------------------------------------------

def get_db_path(corpus: str, n: int) -> str:
    mapping = {
        "livres": f"/opt/bazoulay/ngram/{n}gram.db",
        "presse": f"/opt/bazoulay/ngram/{n}gram_presse.db",
        "lemonde": f"/opt/bazoulay/ngram/{n}gram_lemonde.db",
        "huma": f"/opt/bazoulay/ngram/{n}gram_huma.db",
        "figaro": f"/opt/bazoulay/ngram/{n}gram_figaro.db",
        "moniteur": f"/opt/bazoulay/ngram/{n}gram_moniteur.db",
        "paris": f"/opt/bazoulay/ngram/{n}gram_paris.db",
        "temps": f"/opt/bazoulay/ngram/{n}gram_temps.db",
        "petit_journal": f"/opt/bazoulay/ngram/{n}gram_petit_journal.db",
        "journal_des_debats": f"/opt/bazoulay/ngram/{n}gram_journal_des_debats.db",
        "ddb": f"/opt/bazoulay/ngram/{n}gram_ddb.db"
    }
    return mapping.get(corpus, f"/opt/bazoulay/ngram/{n}gram_{corpus}.db")

@contextlib.contextmanager
def get_db_conn(corpus: str, n: int):
    """Générateur de contexte pour manipuler proprement la base SQLite"""
    path = get_db_path(corpus, n)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_COLNAMES)
    try:
        yield conn
    finally:
        conn.close()

def get_base(corpus: str, n: int) -> pd.DataFrame:
    if corpus == "lemonde":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/lemonde{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "ddb":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/ddb{n}.csv")
        base.columns = ['total', 'annee', 'mois']
    elif corpus == "huma":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/humanite{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "moniteur":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/moniteur{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "figaro":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/figaro{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "paris":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/paris{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "temps":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/temps{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "petit_journal":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/petit_journal{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "journal_des_debats":
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/journal_des_debats{n}.csv")
        base.columns = ['total', 'annee', 'mois', 'jour']
    elif corpus == "presse":
        file = "/opt/bazoulay/docker_gallicagram/gallicagram/base_presse_mois_gallica_monogrammes.csv"
        base = pd.read_csv(file)
        base[["annee", "mois"]] = base.date.str.split("/", expand=True)
        base.drop("date", axis=1, inplace=True)
        base = base.astype("int64")
        base.columns = ['total', 'annee', 'mois']
    elif corpus == "livres":
        base = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/base_livres_gallica_monogrammes.csv")
        base.columns = ["annee", "total"]
    else:
        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/{corpus}{n}.csv")
        base = base.rename(columns={"n": "total"})
    return base

def process_input_words(words_input: str):
    """
    Retourne (words_to_search, original_words).
    - words_to_search : termes passes a la requete SQL (inclut les variantes l'xxx et les composants des sommes '+')
    - original_words  : termes originaux saisis (par ex. ['guerre+guerres']), cles du squelette dans process_results
    """
    input_words = [w.strip() for w in words_input.split(',')]
    words_to_search = []
    vowels = 'aeiouàâéèêëîïôùûü'
    for word in input_words:
        # On découpe par le caractère '+' pour extraire les termes individuels à sommer
        components = [c.strip() for c in word.split('+') if c.strip()]
        for comp in components:
            words_to_search.append(comp)
            if comp and comp[0].lower() in vowels:
                words_to_search.append(f"l'{comp}")
                words_to_search.append(f"l\u2019{comp}")  # apostrophe typographique
    return words_to_search, input_words

def build_query(words_to_search: List[str], fr: int, to: int, corpus: str, resolution: str, rubrique: Optional[str], by_rubrique: bool):
    placeholder = ','.join(['?'] * len(words_to_search))
    query_params = words_to_search + [fr, to]

    if resolution == "default" or resolution == "jour" or corpus == "livres" or (resolution == "mois" and corpus in ["presse", "ddb", "lemonde_rubriques", "mediapart", "lefigaro", "lesechos", "leparisien", "lacroix"]):
        query = f"SELECT * FROM gram WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ?"
    elif resolution == "annee":
        base_table = "gram_mois" if corpus in ["libe", "lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien"] else "gram"
        query = f"SELECT sum(n) as n, annee, gram FROM {base_table} WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ? GROUP BY annee, gram"
    elif resolution == "mois" and corpus in ["libe", "lemonde", "huma", "paris", "figaro", "moniteur", "temps", "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien"]:
        query = f"SELECT * FROM gram_mois WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ?"
    else:
        query = f"SELECT * FROM gram WHERE gram IN ({placeholder}) AND annee BETWEEN ? AND ?"

    if corpus == "lemonde_rubriques":
        query, query_params = handle_lemonde_rubriques(query, query_params, rubrique, resolution, by_rubrique)
    return query, query_params

def handle_lemonde_rubriques(query: str, query_params: list, rubrique: Optional[str], resolution: str, by_rubrique: bool):
    if rubrique:
        rubrique_list = rubrique.split()
        if 1 < len(rubrique_list) < 8:
            rubrique_condition = f"AND rubrique IN ({','.join(['?'] * len(rubrique_list))})"
            query_params = query_params[:-2] + rubrique_list + query_params[-2:]
        elif len(rubrique_list) != 8:
            rubrique_condition = "AND rubrique = ?"
            query_params = query_params[:-2] + [rubrique] + query_params[-2:]
        else:
            rubrique_condition = ""
        query = query.replace("AND annee BETWEEN", f'{rubrique_condition} AND annee BETWEEN')

    if by_rubrique and resolution == "annee":
        query = query.replace("SELECT sum(n) as n, annee, gram", "SELECT sum(n) as n, annee, gram, rubrique")
        query = query.replace("GROUP BY annee, gram", "GROUP BY annee, gram, rubrique")
    elif resolution == "mois":
        query = query.replace("SELECT sum(n) as n, annee, gram", "SELECT sum(n) as n, annee, mois, gram")
        query = query.replace("GROUP BY annee, gram", "GROUP BY annee, mois, gram")

    return query, query_params


# ---------------------------------------------------------------------------
# CONSTRUCTION DU RÉFÉRENTIEL TEMPOREL COMPLET (sans trous)
# ---------------------------------------------------------------------------

_CORPUS_JOURNALIERS = {
    "libe", "lemonde", "huma", "paris", "figaro", "moniteur", "temps",
    "petit_journal", "constitutionnel", "journal_des_debats", "la_presse", "petit_parisien",
}
_CORPUS_MENSUELS = {"presse", "ddb", "mediapart", "lefigaro", "lesechos", "leparisien", "lacroix"}


def _generate_temporal_grid(fr: int, to: int, time_cols: List[str]) -> pd.DataFrame:
    """
    Génère un DataFrame complet de dates sans trous entre 'fr' et 'to' (inclus).
    Utilise 'datetime.date' pour supporter les années antérieures à 1678 (limite de Pandas).
    """
    if "jour" in time_cols:
        start_date = date(fr, 1, 1)
        end_date = date(to, 12, 31)
        delta = end_date - start_date
        
        records = []
        for i in range(delta.days + 1):
            curr = start_date + timedelta(days=i)
            records.append({
                "annee": curr.year,
                "mois": curr.month,
                "jour": curr.day
            })
        grid = pd.DataFrame(records)
        needed_cols = [c for c in time_cols if c in ["annee", "mois", "jour"]]
        return grid[needed_cols]
        
    elif "mois" in time_cols:
        records = []
        for y in range(fr, to + 1):
            for m in range(1, 13):
                records.append({"annee": y, "mois": m})
        return pd.DataFrame(records)
        
    else:
        years = list(range(fr, to + 1))
        return pd.DataFrame({"annee": years})


def _build_base_ref(
    base: pd.DataFrame,
    corpus: str,
    resolution: str,
    rubrique: Optional[str],
    by_rubrique: bool,
    fr: int,
    to: int,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Normalise `base` en un référentiel temporel complet agrégé selon corpus/resolution/rubrique.
    Retourne (base_ref, time_cols) où time_cols sont les colonnes de jointure temporelles.
    """
    # 1. Identifier les colonnes temporelles nécessaires
    if corpus == "livres":
        temporal_cols = ["annee"]
    elif corpus == "lemonde_rubriques":
        temporal_cols = ["annee"]
        if resolution == "mois":
            temporal_cols.append("mois")
    elif resolution == "annee":
        temporal_cols = ["annee"]
    elif resolution == "mois" and (corpus in _CORPUS_JOURNALIERS or corpus in _CORPUS_MENSUELS):
        temporal_cols = ["annee", "mois"]
    else:
        possible = ["annee", "mois", "jour"]
        temporal_cols = [c for c in possible if c in base.columns]
        if not temporal_cols:
            temporal_cols = ["annee"]

    # 2. Générer la grille temporelle complète (sans rupture temporelle)
    temporal_grid = _generate_temporal_grid(fr, to, temporal_cols)

    # 3. Gérer la dimension rubrique si besoin
    if corpus == "lemonde_rubriques":
        if rubrique:
            active_rubriques = rubrique.split()
            base_filtered = base[np.isin(base["rubrique"], active_rubriques)]
        else:
            base_filtered = base
            active_rubriques = base_filtered["rubrique"].dropna().unique().tolist() if "rubrique" in base_filtered.columns else []

        if by_rubrique:
            rubriques_df = pd.DataFrame({"rubrique": active_rubriques})
            grid = temporal_grid.merge(rubriques_df, how="cross")
            time_cols = temporal_cols + ["rubrique"]
        else:
            grid = temporal_grid
            time_cols = temporal_cols

        grouping = temporal_cols + (["rubrique"] if by_rubrique else [])
        base_agg = base_filtered.groupby(grouping).agg({"total": "sum"}).reset_index()

    else:
        grid = temporal_grid
        time_cols = temporal_cols
        base_agg = base.groupby(time_cols).agg({"total": "sum"}).reset_index()

    # 4. Jointure gauche : les périodes sans volume dans 'base' reçoivent total = 0
    base_ref = grid.merge(base_agg, on=time_cols, how="left")
    base_ref["total"] = base_ref["total"].fillna(0).astype(int)

    return base_ref, time_cols


def process_results(
    db_df: pd.DataFrame,
    base: pd.DataFrame,
    corpus: str,
    resolution: str,
    rubrique: Optional[str],
    by_rubrique: bool,
    words_searched: List[str],
    original_words: Optional[List[str]] = None,
    fr: int = 1789,
    to: int = 2022,
) -> pd.DataFrame:
    """
    Construit une série temporelle COMPLÈTE (sans trous) pour chaque gram.
    """
    base_ref, time_cols = _build_base_ref(base, corpus, resolution, rubrique, by_rubrique, fr, to)

    # Construire le mapping variante -> terme original
    skeleton_grams: List[str] = original_words if original_words else words_searched
    variant_map: dict = {}
    for orig in skeleton_grams:
        # On découpe la formule d'origine par le '+' pour mapper chaque composant et ses variantes élidées à celle-ci
        components = [c.strip() for c in orig.split('+') if c.strip()]
        for comp in components:
            variant_map[comp] = orig
            variant_map[f"l'{comp}"] = orig
            variant_map[f"l\u2019{comp}"] = orig  # apostrophe typographique
        variant_map[orig] = orig

    # Normaliser la colonne gram de db_df
    if "gram" in db_df.columns:
        db_df = db_df.copy()
        db_df["gram"] = db_df["gram"].map(lambda g: variant_map.get(str(g), str(g)))
    else:
        db_df = db_df.copy()
        db_df["gram"] = skeleton_grams[0] if skeleton_grams else ""

    # Produit cartésien gram × référentiel temporel sans trous
    grams_df = pd.DataFrame({"gram": skeleton_grams})
    skeleton = base_ref.merge(grams_df, how="cross")

    # Clés de jointure
    merge_keys = [c for c in time_cols if c != "rubrique"] + ["gram"]
    if corpus == "lemonde_rubriques" and by_rubrique and "rubrique" in base_ref.columns:
        merge_keys = time_cols + ["gram"]

    # Agréger db_df sur les clés (somme des éléments de l'addition)
    agg_cols = merge_keys + ["n"]
    db_sub = db_df[[c for c in agg_cols if c in db_df.columns]].copy()
    db_sub = db_sub.groupby(merge_keys, as_index=False).agg({"n": "sum"})

    result = skeleton.merge(db_sub, on=merge_keys, how="left")
    result["n"] = result["n"].fillna(0).astype(int)

    if corpus == "livres":
        result = result.sort_values("annee")

    return result


# ---------------------------------------------------------------------------
# FORMATAGE TEMPOREL ET SÉRIALISATION COLONNAIRE
# ---------------------------------------------------------------------------

def format_temporal_to_iso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les colonnes 'annee', 'mois', 'jour' par une unique colonne 'date'
    au format ISO 'YYYY-MM-DD'.
    """
    if df.empty:
        return df

    years = df["annee"].fillna(0).astype(int).astype(str)

    if "jour" in df.columns and df["jour"].notna().any():
        months = df["mois"].fillna(1).astype(int).astype(str).str.zfill(2)
        days = df["jour"].fillna(1).astype(int).astype(str).str.zfill(2)
        df["date"] = years + "-" + months + "-" + days
    elif "mois" in df.columns and df["mois"].notna().any():
        months = df["mois"].fillna(1).astype(int).astype(str).str.zfill(2)
        df["date"] = years + "-" + months + "-01"
    else:
        df["date"] = years + "-01-01"

    cols_to_drop = [c for c in ["annee", "mois", "jour"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df

def to_columnar_series(df: pd.DataFrame, group_cols: List[str] = ["gram"]) -> List[Dict[str, Any]]:
    """
    Transforme un DataFrame plat en listes colonnaires par groupe.
    Calcule vectoriellement la fréquence relative freq = n / total.
    """
    if df.empty:
        return []

    df = df.replace({np.nan: None})
    df["freq"] = np.where(df["total"] > 0, df["n"] / df["total"], 0.0)

    grouped_results = []

    for keys, group_df in df.groupby(group_cols):
        group_df = group_df.sort_values("date")

        item = {
            "dates": group_df["date"].tolist(),
            "n": [int(x) for x in group_df["n"].tolist()],
            "total": [int(x) for x in group_df["total"].tolist()],
            "freq": [float(x) for x in group_df["freq"].tolist()],
        }

        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(group_cols, keys_tuple):
            item[col] = str(val) if val is not None else None

        grouped_results.append(item)

    return grouped_results

def to_clean_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Gère le typage strict pour éliminer les NaN incompatibles avec JSON (cas non-temporels)"""
    df = df.replace({np.nan: None})
    records = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                clean_row[k] = None
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = int(v) if v == int(v) else float(v)
            else:
                clean_row[k] = v
        records.append(clean_row)
    return records


# ---------------------------------------------------------------------------
# LOGIQUE PARTAGÉE DES REQUÊTES TEMP-SERIES
# ---------------------------------------------------------------------------

def perform_query(
    mot: str,
    corpus: str,
    fr: int,
    to: int,
    resolution: str,
    rubrique: Optional[str],
    by_rubrique: bool
) -> List[Dict[str, Any]]:
    """Exécute et formate la requête SQL et le traitement des n-grammes"""
    words_input = mot.lower()
    words_to_search, original_words = process_input_words(words_input)
    
    # Calcul de la longueur maximale du n-gramme parmi tous les composants
    components_word_counts = []
    for term in words_input.split(','):
        for comp in term.split('+'):
            comp_clean = comp.strip()
            if comp_clean:
                components_word_counts.append(len(comp_clean.split()))
    n = max(components_word_counts) if components_word_counts else 1
    
    if corpus == "prenoms":
        n = 1

    with get_db_conn(corpus, n) as conn:
        query_str, query_params = build_query(words_to_search, fr, to, corpus, resolution, rubrique, by_rubrique)
        db_df = pd.read_sql_query(query_str, conn, params=query_params)

    base = get_base(corpus, n)
    base = base[(base.annee >= int(fr)) & (base.annee <= int(to))]

    db_df = process_results(
        db_df, base, corpus, resolution, rubrique, by_rubrique,
        words_searched=words_to_search,
        original_words=original_words,
        fr=fr,
        to=to
    )

    db_df = format_temporal_to_iso(db_df)
    group_cols = ["gram", "rubrique"] if (by_rubrique and "rubrique" in db_df.columns) else ["gram"]
    return to_columnar_series(db_df, group_cols=group_cols)


# ---------------------------------------------------------------------------
# ENDPOINTS API V2 ENRICHIS
# ---------------------------------------------------------------------------

@app.get(
    "/v2/query", 
    response_model=OptimizedQueryResponse,
    summary="Séries temporelles d'usage de mots (Fréquences et occurrences)",
    description="""
Outil principal pour mesurer l'évolution de la fréquence d'usage d'un ou plusieurs mots dans le temps.

### Fonctionnalités et syntaxe du paramètre `mot` :
- **Recherche simple** : un mot ou syntagme (ex: `"cheval"`, `"chemin de fer"`).
- **Comparaison de termes** : séparez les termes par des virgules pour obtenir des séries distinctes (ex: `"bicyclette,automobile,cheval"`).
- **Somme de variantes / lemmatisation** : utilisez le signe `+` pour additionner des formes fléchies ou synonymes en une seule série (ex: `"cheval+chevaux"`, `"guerre+guerres"`).
- **Élision automatique** : pour les mots commençant par une voyelle, les formes élidées (`l'amour`, `l’amour`) sont automatiquement incluses.

### Corpus disponibles (`corpus`) :
- `presse` (défaut, 1789–1950, n-gram max 3, résolutions: `annee`, `mois`) : Presse française numérisée de Gallica.
- `livres` (1600–1940, n-gram max 5, résolution: `annee`) : Livres français numérisés de Gallica.
- `lemonde` (1944–2023, n-gram max 4, résolutions: `annee`, `mois`, `jour`) : Quotidien Le Monde.
- `lemonde_rubriques` (1944–2023, n-gram max 4) : Le Monde ventilé ou filtré par rubriques (ex: `international`, `politique`, `societe`, `economie`, etc.).
- Journaux historiques (résolutions: `annee`, `mois`, `jour`) : `figaro` (1854-1952), `huma` (1904-1952), `temps` (1861-1942), `moniteur` (1789-1869), `paris` (1777-1827), `journal_des_debats` (1789-1944), `petit_journal` (1863-1942), `petit_parisien` (1876-1944), `la_presse` (1836-1869), `constitutionnel` (1821-1913).
- Autres : `ddb` (Presse allemande 1780-1950), `subtitles` (Sous-titres FR 1935-2020), `subtitles_en` (Sous-titres EN 1930-2020), `rap` (Paroles de rap FR 1989-2024).

### Données renvoyées :
Format colonnaire optimisé contenant pour chaque terme : `dates` (format ISO YYYY-MM-DD), `n` (nombre d'occurrences absolues), `total` (volume total de mots dans le corpus à cette date), `freq` (fréquence relative normalisée = n / total).
"""
)
async def query_v2(
    mot: str = Query(
        ..., 
        description="Mot, syntagme ou ensemble de termes. Utilisez ',' pour comparer plusieurs termes (ex: 'vélo,auto') et '+' pour sommer des variantes (ex: 'cheval+chevaux').",
        examples=["cheval+chevaux,automobile"]
    ),
    corpus: str = Query(
        "presse", 
        description=CORPUS_DESCRIPTION,
        examples=["lemonde"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    resolution: str = Query(
        "default", 
        description="Granularité temporelle : 'annee' (recommandé), 'mois', ou 'jour' (selon corpus).", 
        enum=["default", "jour", "mois", "annee"],
        examples=["annee"]
    ),
    rubrique: Optional[str] = Query(
        None, 
        description="**Corpus `lemonde_rubriques` uniquement.** Codes des rubriques séparés par un espace (ex: 'international economie').",
        examples=["international politique"]
    ),
    by_rubrique: bool = Query(
        False, 
        description="**Corpus `lemonde_rubriques` uniquement.** Si True, sépare et ventile les séries temporelles par rubrique."
    )
):
    """
    Récupère les séries temporelles de fréquence d'usage et d'occurrences pour un ou plusieurs mots/syntagmes.
    Supporte la comparaison (séparateur ',') et la sommation de variantes (séparateur '+').
    """
    try:
        grouped_data = perform_query(mot, corpus, fr, to, resolution, rubrique, by_rubrique)
        return {
            "corpus": corpus,
            "resolution": resolution,
            "results": grouped_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/chart")
async def chart_v2(
    mot: str = Query(
        ..., 
        description="Mot ou expression recherchée. Plusieurs termes peuvent être séparés par des virgules.",
        examples=["patate"]
    ),
    corpus: str = Query(
        "presse", 
        description=CORPUS_DESCRIPTION,
        examples=["lemonde"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (format YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (format YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    resolution: str = Query(
        "default", 
        description="Granularité temporelle.", 
        enum=["default", "jour", "mois", "annee"],
        examples=["annee"]
    ),
    rubrique: Optional[str] = Query(
        None, 
        description="**Corpus `lemonde_rubriques` uniquement.** Codes des rubriques à filtrer, séparés par un espace.",
        examples=["international politique"]
    ),
    by_rubrique: bool = Query(
        False, 
        description="**Corpus `lemonde_rubriques` uniquement.** Si `True`, sépare les séries du graphique par rubrique."
    ),
    chart_type: str = Query(
        "line", 
        description="Type de graphique à générer : 'line' (courbe) ou 'bar' (barres verticales).", 
        enum=["line", "bar"]
    ),
    smoothing: Optional[int] = Query(
        None, 
        description="Fenêtre de lissage (moyenne mobile centrée). Optionnel.", 
        ge=1
    )
):
    if chart_type not in ["line", "bar"]:
        raise HTTPException(status_code=400, detail="Le paramètre chart_type doit être 'line' ou 'bar'")

    try:
        # 1. Récupération des données via la fonction mutualisée
        results = perform_query(mot, corpus, fr, to, resolution, rubrique, by_rubrique)
        if not results:
            raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour cette requête.")

        # 2. Restructuration en DataFrame (Valeurs absolues 'n' pour bar, fréquences 'freq' pour line)
        series_dict = {}
        for res in results:
            label = res["gram"]
            if "rubrique" in res and res["rubrique"]:
                label = f"{res['gram']} ({res['rubrique']})"
            
            dates = pd.to_datetime(res["dates"])
            metric = "n" if chart_type == "bar" else "freq"
            series_dict[label] = pd.Series(res[metric], index=dates)

        plot_df = pd.DataFrame(series_dict).sort_index()
        n_rows = len(plot_df)

        # 3. Initialisation de la figure Matplotlib (Résolution poussée à 200 DPI pour plus de netteté)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        
        fig, ax = plt.subplots(figsize=(11, 6), dpi=200, facecolor='white')
        ax.set_facecolor('white')

        # Palette de base demandée (Set1)
        base_colors = [
            '#377EB8FF',  # Bleu
            '#E41A1CFF',  # Rouge
            '#4DAF4AFF',  # Vert
            '#984EA3FF',  # Violet
            '#FF7F00FF',  # Orange
            '#FFFF33FF',  # Jaune
            '#A65628FF',  # Marron
            '#F781BFFF',  # Rose
            '#999999FF'   # Gris
        ]

        # Détermination du nombre de séries à tracer et génération de la palette
        n_series = len(plot_df.columns)
        if n_series <= len(base_colors):
            colors = base_colors[:n_series]
        else:
            # S'il y a plus de 9 séries, on étend dynamiquement avec la colormap tab20
            colors = list(base_colors)
            needed = n_series - len(base_colors)
            try:
                cmap = matplotlib.colormaps['tab20']
            except AttributeError:
                cmap = plt.cm.get_cmap('tab20')
                
            for i in range(needed):
                colors.append(mcolors.to_hex(cmap(i % 20)))

        window_size = 1

        # Détermination du format d'affichage de la date selon la résolution demandée
        if resolution == "annee":
            date_fmt = '%Y'
        elif resolution == "mois":
            date_fmt = '%Y-%m'
        elif resolution == "jour":
            date_fmt = '%Y-%m-%d'
        else:  # "default" - Déduction de la granularité à partir des données réelles
            sample_dates = plot_df.index
            if all(d.month == 1 and d.day == 1 for d in sample_dates):
                date_fmt = '%Y'
            elif all(d.day == 1 for d in sample_dates):
                date_fmt = '%Y-%m'
            else:
                date_fmt = '%Y-%m-%d'

        # Cas d'un graphique linéaire (Line Chart)
        if chart_type == "line":
            # Calcul du lissage automatique si non spécifié
            if smoothing is None:
                if n_rows > 500:
                    window_size = max(5, n_rows // 80)
                elif n_rows > 150:
                    window_size = max(3, n_rows // 40)
                elif n_rows > 50:
                    window_size = 3
                else:
                    window_size = 1
            else:
                window_size = max(1, smoothing)

            # Application du lissage (moyenne mobile centrée)
            if window_size > 1:
                plot_df_smoothed = plot_df.rolling(window=window_size, min_periods=1, center=True).mean()
            else:
                plot_df_smoothed = plot_df

            # Tracé des courbes
            for i, col in enumerate(plot_df_smoothed.columns):
                color = colors[i % len(colors)]
                ax.plot(plot_df_smoothed.index, plot_df_smoothed[col], label=col, color=color, linewidth=2, alpha=0.9)

            # Ajuster les limites de l'axe X pour supprimer l'espace blanc aux extrémités
            if not plot_df_smoothed.empty:
                ax.set_xlim(plot_df_smoothed.index.min(), plot_df_smoothed.index.max())

            # Formatage précis de l'axe X selon la résolution
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
            if date_fmt == '%Y':
                ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, n_rows // 15)))
            elif date_fmt == '%Y-%m':
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, n_rows // 15)))
            else:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.xticks(rotation=30, ha='right')

        # Cas d'un diagramme en barres (Bar Chart)
        elif chart_type == "bar":
            sample_dates = plot_df.index
            x_labels = sample_dates.strftime(date_fmt)
            
            plot_df.index = x_labels
            plot_df.plot(kind='bar', ax=ax, width=0.8, color=colors[:len(plot_df.columns)], edgecolor='none', alpha=0.85)
            
            # Gestion de la densité d'affichage sur l'axe X
            n_ticks = len(x_labels)
            if n_ticks > 25:
                step = max(1, n_ticks // 15)
                ax.set_xticks(range(0, n_ticks, step))
                ax.set_xticklabels(x_labels[::step], rotation=45, ha='right')
            else:
                ax.set_xticklabels(x_labels, rotation=45 if n_ticks > 8 else 0, ha='right' if n_ticks > 8 else 'center')

        # Formatage de l'axe Y et des légendes selon le type de graphique
        if chart_type == "bar":
            # Valeurs absolues (n) -> entiers avec séparateurs de milliers
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
            ax.set_ylabel("Nombre d'occurrences", fontsize=10.5, color='#2C3E50', labelpad=10)
        else:
            # Fréquence relative (%) -> décimales dynamiques
            max_val = plot_df.max().max()
            if max_val > 0:
                if max_val < 0.01:
                    decimals = max(2, int(-math.log10(max_val)) + 2)
                else:
                    decimals = 2
            else:
                decimals = 2
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=decimals))
            ax.set_ylabel("Fréquence relative", fontsize=10.5, color='#2C3E50', labelpad=10)

        # Style des axes et grilles
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#BDC3C7')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')

        # Titres et labels
        title_text = f"Évolution : {', '.join(mot.split(','))}"
        ax.set_title(title_text, fontsize=12.5, fontweight='bold', color='#2C3E50', pad=25)
        
        # Sous-titre contenant le lissage et les informations de métadonnées
        subtitle_text = f"Corpus: {corpus} | Résolution: {resolution} | Période: {fr} - {to}"
        if chart_type == "line" and window_size > 1:
            subtitle_text += f" | Lissage (moyenne mobile) : {window_size} points"
        ax.text(0.5, 1.02, subtitle_text, transform=ax.transAxes, ha='center', fontsize=9.5, color='#7F8C8D')

        # Légende positionnée horizontalement sous le graphique
        n_cols = len(plot_df.columns)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(5, n_cols),
            frameon=True,
            facecolor='white',
            edgecolor='#E5E7E9',
            framealpha=0.9,
            fontsize=9
        )

        plt.tight_layout()

        # Sauvegarde en mémoire et réponse à 200 DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        plt.close('all')
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du graphique: {str(e)}")


@app.get(
    "/v2/contain", 
    response_model=OptimizedQueryResponse,
    summary="Cooccurrence temporelle de 2 termes dans un même n-gramme",
    description="""
Mesure l'évolution temporelle de l'apparition conjointe de deux termes (`mot1` et `mot2`) au sein d'une même fenêtre textuelle étroite (n-gramme de 3 à 4 mots selon le corpus).

### Quand utiliser cet outil :
- Pour analyser des syntagmes complexes ou des collocations sans ordre strict (ex: `mot1="crise"`, `mot2="financière"` ou `mot1="intelligence"`, `mot2="artificielle"`).
- Pour vérifier si deux notions sont associées dans la même phrase ou proposition.

### Paramètre `count` :
- `count=True` (défaut) : renvoie une série temporelle unique agrégeant toutes les combinaisons trouvées.
- `count=False` : renvoie les séries détaillées pour chaque n-gramme distinct contenant les deux mots.
"""
)
async def contain_v2(
    mot1: str = Query(
        ..., 
        description="Premier mot ou lemme obligatoire dans le n-gramme.",
        examples=["crise"]
    ),
    mot2: str = Query(
        ..., 
        description="Second mot ou lemme obligatoire dans le n-gramme.",
        examples=["financière"]
    ),
    corpus: str = Query(
        "lemonde", 
        description=CORPUS_DESCRIPTION,
        examples=["lemonde"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    count: bool = Query(
        True, 
        description="Si True, agrège en une seule série temporelle globale. Si False, détaille par n-gramme trouvé."
    ),
    resolution: str = Query(
        "default", 
        description="Granularité temporelle des résultats.", 
        enum=["default", "jour", "mois", "annee"],
        examples=["annee"]
    )
):
    """
    Recherche l'évolution temporelle de n-grammes contenant obligatoirement deux mots (mot1 ET mot2) dans la même fenêtre textuelle.
    """
    mot1_clean = mot1.replace("'", "").lower()
    mot2_clean = mot2.replace("'", "").lower()
    base_table = "gram"
    gram_label = f"{mot1_clean}&{mot2_clean}"

    if corpus == "presse":
        n = 3
        time_steps = "annee,mois"
    elif corpus == "livres":
        n = 3
        time_steps = "annee"
    elif corpus == "lemonde":
        n = 4
        time_steps = "annee,mois"
        base_table = "gram_mois"
    else:
        n = 3
        time_steps = "annee"

    try:
        with get_db_conn(corpus, n) as conn:
            if count:
                sql = (
                    f"SELECT sum(n) as n,{time_steps} FROM {base_table} "
                    f"WHERE rowid IN (SELECT rowid FROM full_text WHERE gram MATCH '{mot1_clean} AND {mot2_clean}') "
                    f"AND annee BETWEEN {fr} AND {to} GROUP BY {time_steps}"
                )
            else:
                sql = (
                    f"SELECT sum(n) as n,gram,{time_steps} FROM {base_table} "
                    f"WHERE rowid IN (SELECT rowid FROM full_text WHERE gram MATCH '{mot1_clean} AND {mot2_clean}') "
                    f"AND annee BETWEEN {fr} AND {to} GROUP BY gram,{time_steps}"
                )
            db_df = pd.read_sql_query(sql, conn)

        if "gram" not in db_df.columns:
            db_df["gram"] = gram_label

        base = get_base(corpus, n)
        base = base.loc[(base.annee >= int(fr)) & (base.annee <= int(to))]

        db_df = process_results(
            db_df, base, corpus, resolution,
            rubrique=None, by_rubrique=False,
            words_searched=[gram_label],
            fr=fr,
            to=to
        )

        db_df = format_temporal_to_iso(db_df)
        grouped_data = to_columnar_series(db_df)

        return {
            "corpus": corpus,
            "resolution": resolution,
            "results": grouped_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v2/joker", 
    response_model=List[TermFrequency],
    summary="Exploration lexicale - Mots apparaissant autour d'un terme (Wildcard)",
    description="""
Recherche exploratoire (Wildcard / joker) découvrant les mots ou syntagmes les plus fréquents apparaissant immédiatement avant ou après un mot pivot.

### Quand utiliser cet outil :
- Découvrir des adjectifs ou compléments fréquents (ex: que trouve-t-on après `"société"` ou après `"république"` ?).
- Découvrir des verbes, déterminants ou titres (ex: que trouve-t-on avant `"pasteur"` ou `"napoléon"` ?).
- Explorer des formules figées d'une époque donnée.

### Paramètres clés :
- `after=True` (défaut) : cherche les mots qui SUIVENT le mot pivot (ex: `mot="chambre"` -> `"des députés"`, `"syndicale"`).
- `after=False` : cherche les mots qui PRÉCÈDENT le mot pivot (ex: `mot="guerre"` -> `"grande"`, `"première"`, `"sainte"`).
- `n_joker` : nombre maximal de termes à renvoyer (ex: `10`, `50` ou `"all"`).
- `length` : taille totale du n-gramme (pivot inclus). Par exemple, `length=2` renvoie le mot adjacent immédiat (bigramme).
"""
)
async def joker_v2(
    mot: str = Query(
        ..., 
        description="Mot pivot autour duquel rechercher les compléments lexicaux.",
        examples=["camarade"]
    ),
    corpus: str = Query(
        "lemonde", 
        description=CORPUS_DESCRIPTION,
        examples=["lemonde"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    after: bool = Query(
        True, 
        description="Si True, cherche les mots qui SUIVENT le mot pivot. Si False, cherche les mots qui PRÉCÈDENT."
    ),
    n_joker: str = Query(
        "50", 
        description="Nombre maximal de résultats à renvoyer (entier ou 'all').",
        examples=["10"]
    ),
    length: Optional[int] = Query(
        None, 
        description="Longueur du n-gramme total (pivot inclus). 2 = mot adjacent immédiat, 3 = 2 mots adjacents.",
        ge=1,
        le=5,
        examples=[2]
    )
):
    """
    Recherche les termes apparaissant le plus fréquemment immédiatement avant ou après un mot pivot (recherche joker/wildcard).
    """
    mot_clean = mot.lower()
    n = length if length is not None else len(mot_clean.split(" ")) + 1
    base_table = "gram_mois" if corpus == "lemonde" else "gram"
    limit = "" if n_joker == "all" else f"limit {n_joker}"

    try:
        with get_db_conn(corpus, n) as conn:
            if after:
                sql = (
                    f"select sum(n) as tot, gram from {base_table} "
                    f"where annee between {fr} and {to} "
                    f"and rowid in (select rowid from full_text where gram match '^ {mot_clean}') "
                    f"group by gram order by tot desc {limit}"
                )
            else:
                sql = (
                    f"select sum(n) as tot, gram from {base_table} "
                    f"where annee between {fr} and {to} "
                    f"and rowid in (select rowid from full_text where gram match '{mot_clean}') "
                    f"group by gram order by tot desc {limit}"
                )
            db_df = pd.read_sql_query(sql, conn)

        return to_clean_json_records(db_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v2/associated", 
    response_model=List[TermFrequency],
    summary="Voisinage lexical et cooccurrences locales (Fenêtre n-gramme)",
    description="""
Identifie les mots les plus fréquemment associés dans le voisinage immédiat d'un mot cible (collocations au sein des n-grammes), avec filtrage des mots vides (stopwords).

### Quand utiliser cet outil :
- Analyse sémantique et contextuelle d'un mot à une époque donnée.
- Découvrir l'univers lexical ou discursif associé à une notion (ex: quels termes entourent `"progrès"` ou `"atome"` entre 1900 et 1950 ?).

### Paramètre `stopwords` :
- `stopwords=0` : conserve tous les mots.
- `stopwords=500` (recommandé) : élimine les 500 mots les plus fréquents de la langue française (`de`, `le`, `la`, `et`, `en`, etc.) pour ne faire ressortir que les termes signifiants.
"""
)
async def associated_v2(
    mot: str = Query(
        ..., 
        description="Mot cible pour lequel extraire le voisinage lexical.",
        examples=["changement"]
    ),
    corpus: str = Query(
        "lemonde", 
        description=CORPUS_DESCRIPTION,
        examples=["lemonde"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    n_joker: str = Query(
        "50", 
        description="Nombre maximal de voisins lexicaux à renvoyer.",
        examples=["10"]
    ),
    length: Optional[int] = Query(
        None, 
        description="Largeur de la fenêtre de n-gramme (max 3 pour Gallica, max 4 pour Le Monde).",
        ge=1,
        le=5,
        examples=[2]
    ),
    stopwords: int = Query(
        0, 
        description="Nombre de mots les plus fréquents de la langue française à filtrer (ex: 500 pour éliminer les mots grammaticaux).",
        ge=0,
        le=1000,
        examples=[500]
    )
):
    """
    Extrait les mots les plus fréquemment associés dans le voisinage immédiat d'un mot cible, avec filtrage optionnel des mots vides.
    """
    mot_clean = mot.lower()
    n = length if length is not None else len(mot_clean.split(" ")) + 1
    base_table = "gram_mois" if corpus == "lemonde" else "gram"

    try:
        with get_db_conn(corpus, n) as conn:
            sql = (
                f'select sum(n) as tot, gram from {base_table} '
                f'where annee between {fr} and {to} '
                f'and rowid in (select rowid from full_text where gram match "{mot_clean}") '
                f'group by gram order by tot desc'
            )
            db_df = pd.read_sql_query(sql, conn)

        z = db_df.gram.str.endswith(f"{mot_clean}")
        zz = db_df.gram.str.match(f"{mot_clean}")
        db_df = db_df.loc[z + zz]
        db_df["gram"] = db_df.gram.str.split(" ")
        db_df = db_df.explode("gram")

        limit_val = len(db_df.index) if n_joker == "all" else int(n_joker)

        z_filter = [db_df.gram.values[i] not in mot_clean.split(" ") for i in range(len(db_df.index))]
        db_df = (
            db_df.loc[z_filter]
            .groupby("gram")
            .agg({"tot": "sum"})
            .sort_values("tot", ascending=False)
            .reset_index()
        )

        if stopwords > 0:
            sw = pd.read_csv("/opt/bazoulay/docker_gallicagram/gallicagram/stopwords.csv").iloc[:stopwords]
            db_df = db_df.loc[~db_df.gram.isin(sw.monogram)]

        db_df = db_df.iloc[:limit_val]
        return to_clean_json_records(db_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v2/cooccur", 
    response_model=OptimizedQueryResponse,
    summary="Cooccurrence à l'échelle de l'article complet (Le Monde uniquement)",
    description="""
Mesure le nombre et la proportion d'articles du journal *Le Monde* (1944–2023) qui contiennent simultanément deux termes (`mot1` et `mot2`), quelle que soit la distance entre ces termes dans l'article.

### Différence fondamentale avec `/v2/contain` :
- `/v2/contain` cherche dans une fenêtre très étroite de quelques mots consécutifs (n-gramme).
- `/v2/cooccur` cherche dans l'ensemble de l'article de presse (cooccurrence thématique globale).

### Quand l'utiliser :
- Pour analyser si deux sujets ou entités sont traités ensemble dans l'actualité (ex: corrélation entre `"climat"` et `"économie"`, `"ukraine"` et `"otan"`, `"inflation"` et `"salaires"`).
- Possibilité d'inclure des variantes pour chaque terme en les séparant par un espace (ex: `mot1="climatique climatiques"` et `mot2="crise catastrophes"`).
"""
)
async def cooccur_v2(
    mot1: str = Query(
        ..., 
        description="Premier terme ou liste de variantes séparées par un espace (ex: 'climatique climatiques').",
        examples=["climatique climatiques"]
    ),
    mot2: str = Query(
        ..., 
        description="Second terme ou liste de variantes séparées par un espace (ex: 'crise catastrophes').",
        examples=["crise catastrophes"]
    ),
    fr: int = Query(
        1945, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1945]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    resolution: str = Query(
        "jour", 
        description="Granularité temporelle : 'jour', 'mois' ou 'annee'.", 
        enum=["jour", "mois", "annee"],
        examples=["jour"]
    )
):
    """
    Mesure la cooccurrence de deux mots au sein d'un même article de presse du quotidien Le Monde (1944-2023).
    """
    mot1_clean = mot1.replace("'", "").replace(" ", "','").lower()
    mot2_clean = mot2.replace("'", "").replace(" ", "','").lower()
    gram_label = f"{mot1_clean}&{mot2_clean}"

    time_steps = "annee"
    if resolution in ["mois", "jour"]:
        time_steps += ",mois"
    if resolution == "jour":
        time_steps += ",jour"

    try:
        conn = sqlite3.connect("/opt/bazoulay/ngram/1gram_lemonde_article.db")
        query = (
            f"select {time_steps},count(article_id) as n from ("
            f"select distinct article_id,{time_steps} from gram where gram in ('{mot1_clean}') and annee between {fr} and {to} "
            f"INTERSECT "
            f"select distinct article_id,{time_steps} from gram where gram in ('{mot2_clean}') and annee between {fr} and {to}"
            f") group by {time_steps}"
        )
        db_df = pd.read_sql(query, conn)
        conn.close()

        db_df["gram"] = gram_label

        base = pd.read_csv("/opt/bazoulay/ngram/base_articles.csv")
        base = base.loc[(base.annee >= int(fr)) & (base.annee <= int(to))]

        db_df = process_results(
            db_df, base, corpus="lemonde_article", resolution=resolution,
            rubrique=None, by_rubrique=False,
            words_searched=[gram_label],
            fr=fr,
            to=to
        )

        db_df = format_temporal_to_iso(db_df)
        grouped_data = to_columnar_series(db_df)

        return {
            "corpus": "lemonde_article",
            "resolution": resolution,
            "results": grouped_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v2/query_persee", 
    response_model=PerseeOptimizedQueryResponse,
    summary="Recherche temporelle dans les revues universitaires de sciences humaines (Persée)",
    description="""
Interroge le portail académique *Persée* (1789–2023) regroupant les revues scientifiques et universitaires francophones en sciences humaines et sociales.

### Quand l'utiliser :
- Pour l'histoire des concepts, les sciences sociales, la linguistique ou la philosophie.
- Mesurer la diffusion d'un terme dans la recherche académique plutôt que dans la presse généraliste.

### Paramètres clés :
- `mot` : mot ou syntagme (max 2 mots). Supporte les sommes `+` (ex: `"inégalité+inégalités"`).
- `revue` : code de la revue scientifique (ex: `"arss"` pour Actes de la recherche en sciences sociales, `"ahess"` pour Annales HSS) ou `"all"` pour l'ensemble des revues.
- `by_revue=True` : sépare et compare les séries temporelles par revue d'origine.
"""
)
async def query_persee_v2(
    mot: str = Query(
        ..., 
        description="Mot ou syntagme (max 2 mots). Utilisez '+' pour sommer des variantes (ex: 'sociologie+sociologique').",
        examples=["inégalités"]
    ),
    revue: str = Query(
        "all", 
        description="Codes des revues à filtrer, séparés par un espace (ex: 'arss ahess') ou 'all' pour l'ensemble.",
        examples=["arss ahess"]
    ),
    fr: int = Query(
        1789, 
        alias="from", 
        description="Année de début (YYYY).",
        ge=1600,
        le=2025,
        examples=[1789]
    ),
    to: int = Query(
        2022, 
        description="Année de fin (YYYY).",
        ge=1600,
        le=2025,
        examples=[2022]
    ),
    by_revue: bool = Query(
        False, 
        description="Si True, répartit et compare les résultats par revue scientifique d'origine."
    )
):
    """
    Récupère l'évolution temporelle de termes dans les revues académiques de sciences humaines du portail Persée (1789-2023).
    """
    word = mot.lower()
    components = [c.strip() for c in word.split('+') if c.strip()]
    n = max(len(comp.split()) for comp in components) if components else 1
    revue_condition = ""

    if revue != "all" and len(revue.split(' ')) < 362:
        if " " in revue:
            revue_condition = f"and revue in {tuple(revue.split(' '))}"
        else:
            revue_condition = f'and revue="{revue}"'

    try:
        conn = sqlite3.connect(f"/opt/bazoulay/ngram/{n}gram_persee.db")
        placeholders = ','.join(['?'] * len(components))

        if by_revue:
            query = f'SELECT n,annee,gram,revue from gram where gram IN ({placeholders}) {revue_condition} and annee between {fr} and {to}'
            db_df = pd.read_sql_query(query, conn, params=components)
        else:
            query = f'SELECT sum(n) as n,annee,gram from gram where gram IN ({placeholders}) {revue_condition} and annee between {fr} and {to} group by annee, gram'
            db_df = pd.read_sql_query(query, conn, params=components)
        conn.close()

        base = pd.read_csv(f"/opt/bazoulay/docker_gallicagram/gallicagram/persee{n}.csv")
        base.columns = ["total", "annee", "revue"]
        if revue != "all":
            base = base.loc[np.isin(base.revue, revue.split(" "))]
        base = base.loc[(base.annee >= int(fr)) & (base.annee <= int(to))]

        # 1. Générer la grille temporelle complète pour Persée (annuelle)
        grid = pd.DataFrame({"annee": list(range(fr, to + 1))})

        # 2. Gérer la revue si by_revue est vrai
        if by_revue and "revue" in base.columns:
            unique_revues = base["revue"].dropna().unique().tolist()
            if unique_revues:
                revues_df = pd.DataFrame({"revue": unique_revues})
                grid = grid.merge(revues_df, how="cross")
                merge_keys = ["annee", "revue"]
            else:
                merge_keys = ["annee"]
        else:
            merge_keys = ["annee"]

        # 3. Agréger les totaux et boucher les trous avec total = 0
        base_agg = base.groupby(merge_keys).agg({"total": "sum"}).reset_index()
        base_ref = grid.merge(base_agg, on=merge_keys, how="left")
        base_ref["total"] = base_ref["total"].fillna(0).astype(int)

        # Normalisation des n-grammes vers la formule combinée
        variant_map = {comp: word for comp in components}
        variant_map[word] = word

        if "gram" in db_df.columns:
            db_df = db_df.copy()
            db_df["gram"] = db_df["gram"].map(lambda g: variant_map.get(str(g), str(g)))
        else:
            db_df = db_df.copy()
            db_df["gram"] = word

        group_cols_persee = ["gram", "revue"] if (by_revue and "revue" in db_df.columns) else ["gram"]

        if by_revue and "revue" in base_ref.columns:
            skeleton = base_ref[["annee", "revue", "total"]].copy()
            skeleton["gram"] = word
            merge_keys = ["annee", "revue", "gram"]
        else:
            skeleton = base_ref[["annee", "total"]].copy()
            skeleton["gram"] = word
            merge_keys = ["annee", "gram"]

        db_sub = db_df[[c for c in merge_keys + ["n"] if c in db_df.columns]].copy()
        
        # Consolidation finale des sous-composants regroupés
        db_sub = db_sub.groupby(merge_keys, as_index=False).agg({"n": "sum"})

        result = skeleton.merge(db_sub, on=merge_keys, how="left")
        result["n"] = result["n"].fillna(0).astype(int)

        result = format_temporal_to_iso(result)
        grouped_data = to_columnar_series(result, group_cols=group_cols_persee)

        return {
            "corpus": "persee",
            "results": grouped_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/v2/source_rap", 
    response_model=List[SourceRapResultItem],
    summary="Concordancier et extraits de paroles de rap francophone (Genius / LRFAF)",
    description="""
Recherche textuelle et concordancier dans la base complète des paroles de rap francophone (1989–2024).

### Ce que renvoie l'outil :
Pour chaque occurrence trouvée : l'artiste, le titre du morceau, l'année, l'URL Genius, le nombre d'occurrences, ainsi qu'un concordancier (`context_left`, `pivot`, `context_right`) permettant d'extraire la citation exacte en contexte.

### Quand l'utiliser :
- Pour illustrer l'usage d'un mot d'argot, néologisme ou référence culturelle par des exemples réels citables.
- Pour étudier le discours et les thèmes du rap français pour une année précise.
"""
)
async def source_rap_v2(
    mot: str = Query(
        ..., 
        description="Mot ou expression exacte recherchée dans les textes de rap.",
        examples=["banlieue"]
    ), 
    year: int = Query(
        ..., 
        description="Année cible de l'analyse (1989-2024).",
        ge=1980,
        le=2026,
        examples=[2018]
    )
):
    """
    Extrait les morceaux de rap, métadonnées et citations en contexte (concordancier) pour un terme et une année donnée.
    """
    word_pattern = r"\b" + mot + r"\b"
    word_pattern = word_pattern.replace(r"|", r"\b|\b")

    try:
        corpus_path = os.path.expanduser("~/LRFAF/corpus.csv")
        corpus_rap = pd.read_csv(corpus_path)

        corpus = corpus_rap.loc[corpus_rap.year == year]
        corpus = corpus.loc[corpus.lyrics.str.contains(word_pattern, case=False, na=False)]

        if corpus.empty:
            return []

        corpus["counts"] = corpus.lyrics.str.count(word_pattern, re.I)
        matchs = [re.search(word_pattern, lyrics, flags=re.IGNORECASE) for lyrics in corpus.lyrics.values]

        corpus["context_left"] = [
            corpus.lyrics.values[i][max(0, matchs[i].start() - 30):(matchs[i].start())] if matchs[i] else ""
            for i in range(len(corpus.index))
        ]
        corpus["pivot"] = [
            corpus.lyrics.values[i][matchs[i].start():matchs[i].end()] if matchs[i] else ""
            for i in range(len(corpus.index))
        ]
        corpus["context_right"] = [
            corpus.lyrics.values[i][max(0, matchs[i].end()):(matchs[i].end() + 30)] if matchs[i] else ""
            for i in range(len(corpus.index))
        ]

        corpus = corpus[["year", "artist", "title", "url", "pageviews", "counts", "context_left", "pivot", "context_right"]]
        corpus.url = "<a href='" + corpus.url + "' target='_blank'>" + corpus.url + "</a>"
        corpus = corpus.sort_values("pageviews", ascending=False)
        corpus = corpus.drop("pageviews", axis=1)
        corpus.year = corpus.year.astype("int")

        return to_clean_json_records(corpus)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))