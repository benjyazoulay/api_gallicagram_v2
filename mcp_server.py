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
    mot: str,
    corpus: str = "presse",
    to: int = 2022,
    fr: Annotated[int, Field(alias="from")] = 1789,
    resolution: str = "default",
    rubrique: Optional[str] = None,
    by_rubrique: bool = False,
    chart_type: str = "line",
    smoothing: Optional[int] = None
) -> Image:
    """
    Génère un graphique (courbe ou barres) représentant l'évolution temporelle de la fréquence d'usage de mots dans Gallica.
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