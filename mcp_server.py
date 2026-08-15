# mcp_server.py
from typing import Annotated, Optional
from pydantic import Field
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from app import app, chart_v2 as fastapi_chart_v2

# Gestion des imports pour s'adapter aux versions récentes de FastMCP (v3.x / v2.x)
try:
    from fastmcp.server.providers.openapi import RouteMap, MCPType
except ImportError:
    try:
        from fastmcp.server.openapi import RouteMap, MCPType
    except ImportError:
        RouteMap, MCPType = None, None

# Middleware de mise en cache de réponses FastMCP
try:
    from fastmcp.server.middleware.caching import ResponseCachingMiddleware, CallToolSettings, ListToolsSettings
    HAS_CACHING = True
except ImportError:
    HAS_CACHING = False

# Force la conversion de toutes les routes GET/POST en "Outils" MCP (Tools) 
# en excluant la route de graphique brut qui renvoie du binaire (PNG)
route_maps = [
    RouteMap(pattern=r"^/v2/chart", mcp_type=MCPType.EXCLUDE),
    RouteMap(mcp_type=MCPType.TOOL)
] if RouteMap and MCPType else None

# Génération du serveur MCP à partir de l'instance FastAPI
mcp = FastMCP.from_fastapi(
    app=app,
    name="Gallicagram V2 MCP",
    route_maps=route_maps
)

# Activation du cache de réponses : les séries historiques Gallicagram étant
# immuables (1789-2023), la mise en cache (1 heure) accélère drastiquement
# les appels répétitifs du LLM et évite les surcharges SQL / pandas.
if HAS_CACHING:
    try:
        mcp.add_middleware(
            ResponseCachingMiddleware(
                call_tool_settings=CallToolSettings(ttl=3600),
                list_tools_settings=ListToolsSettings(ttl=3600)
            )
        )
    except Exception as e:
        print(f"[FastMCP] Info: Cache middleware non initialisé ({e})")

@mcp.tool()
async def generate_chart(
    mot: Annotated[
        str, 
        Field(
            description="Terme(s) à visualiser. Utilisez des virgules pour comparer plusieurs courbes (ex: 'vélo,bicyclette,automobile') ou '+' pour sommer des variantes (ex: 'cheval+chevaux')."
        )
    ],
    corpus: Annotated[
        str, 
        Field(
            description="Corpus cible ('presse', 'livres', 'lemonde', 'lemonde_rubriques', 'figaro', 'huma', 'temps', 'ddb', etc.). Défaut: 'presse'."
        )
    ] = "presse",
    to: Annotated[
        int, 
        Field(
            description="Année de fin (YYYY). Défaut: 2022."
        )
    ] = 2022,
    fr: Annotated[
        int, 
        Field(
            alias="from", 
            description="Année de début (YYYY). Défaut: 1789."
        )
    ] = 1789,
    resolution: Annotated[
        str, 
        Field(
            description="Granularité temporelle : 'annee' (recommandé), 'mois', ou 'jour' (selon corpus). Défaut: 'default'."
        )
    ] = "default",
    rubrique: Annotated[
        Optional[str], 
        Field(
            description="**Corpus `lemonde_rubriques` uniquement.** Codes des rubriques filtrées séparés par un espace (ex: 'international economie')."
        )
    ] = None,
    by_rubrique: Annotated[
        bool, 
        Field(
            description="**Corpus `lemonde_rubriques` uniquement.** Si True, trace une courbe distincte par rubrique."
        )
    ] = False,
    chart_type: Annotated[
        str, 
        Field(
            description="Type de rendu visuel : 'line' (courbes lissées des fréquences relatives) ou 'bar' (histogramme des occurrences absolues)."
        )
    ] = "line",
    smoothing: Annotated[
        Optional[int], 
        Field(
            description="Fenêtre de lissage par moyenne mobile centrée (ex: 3 ou 5). Si non spécifié, un lissage automatique optimal est appliqué."
        )
    ] = None
) -> Image:
    """
    Génère un graphique visuel haute résolution (image PNG à 200 DPI) représentant l'évolution temporelle de mots dans Gallicagram.

    ### Quand utiliser cet outil :
    - Quand l'utilisateur demande explicitement un graphique, une visualisation, une courbe ou un tracé visuel.
    - Idéal pour comparer visuellement plusieurs termes ou illustrer une dynamique historique.

    ### Syntaxe des termes (`mot`) :
    - Comparaison : `"mot1,mot2,mot3"` -> plusieurs courbes colorées distinctes avec légende.
    - Somme / Lemmatisation : `"mot1+mot2"` -> une seule courbe combinant les occurrences.

    ### Retour :
    Retourne un objet Image PNG encodé de manière compatible avec les protocoles MCP / Claude Desktop.
    """
    # Appel direct du endpoint FastAPI d'origine
    response = await fastapi_chart_v2(
        mot=mot,
        corpus=corpus,
        fr=fr,
        to=to,
        resolution=resolution,
        rubrique=rubrique,
        by_rubrique=by_rubrique,
        chart_type=chart_type,
        smoothing=smoothing
    )
    # Retourne les données binaires décodées proprement par FastMCP comme contenu image (PNG)
    img = Image(data=response.body, format="png")
    return img.to_image_content() if hasattr(img, "to_image_content") else img

# Convertit l'instance MCP en application ASGI compatible uvicorn
mcp_app = mcp.http_app()

if __name__ == "__main__":
    # Exécution directe en mode Streamable HTTP standard de FastMCP sur le port 8003
    mcp.run(transport="http", host="0.0.0.0", port=8003)