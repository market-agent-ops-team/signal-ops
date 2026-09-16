from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from pydantic import ValidationError

from agents.researcher_agent import TICKER_MAP, get_company_name
from api.routes import AnalyzeRequest, analyze
from ml.experiments.config import EXTENDED_TICKERS
from ml.features.indicators import V1_HISTORY_START


ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=ROOT / "templates")
markdown = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table").disable(["image", "lheading"])
TICKER_SUGGESTIONS = sorted(
    {(symbol, get_company_name(symbol)) for symbol in
     [key.removesuffix(".NS") for key in TICKER_MAP] + list(EXTENDED_TICKERS)}
)


EXAMPLES = {
    "bullish-alignment": {"title": "Bullish alignment", "file": "bullish_alignment.md", "number": "01",
                          "description": "When technical direction and news sentiment both point up."},
    "bearish-alignment": {"title": "Bearish alignment", "file": "bearish_alignment.md", "number": "02",
                          "description": "When both research branches point in a bearish direction."},
    "divergence": {"title": "Signal divergence", "file": "divergence.md", "number": "03",
                   "description": "When the model and the headlines tell different stories."},
}


def render(request: Request, name: str, *, status_code: int = 200, **context: Any):
    preference = request.cookies.get("theme", "system")
    if preference not in {"light", "dark", "system"}:
        preference = "system"
    return templates.TemplateResponse(
        request=request, name=name,
        context={"theme": preference, "return_to": request.url.path + (f"?{request.url.query}" if request.url.query else ""),
                 "ticker": "", "target_date": "", "errors": {}, "examples": EXAMPLES,
                 "ticker_suggestions": TICKER_SUGGESTIONS, "date_min": V1_HISTORY_START,
                 "date_max": date.today().isoformat(), **context},
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Vary": "Cookie",
                 "Content-Security-Policy": "default-src 'self'; script-src 'none'; style-src 'self'; font-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
                 "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin"},
    )


def render_report(source: str) -> str:
    tokens = markdown.parse(source)
    for token in tokens:
        if token.type in {"heading_open", "heading_close"}:
            token.tag = f"h{min(int(token.tag[1]) + 1, 6)}"
    return markdown.renderer.render(tokens, markdown.options, {})


@router.get("/")
async def workspace(request: Request):
    return render(request, "workspace.html", title="Workspace", active="workspace")


@router.get("/report")
async def report(request: Request, ticker: str = "", target_date: str = ""):
    try:
        analysis_request = AnalyzeRequest(ticker=ticker, target_date=target_date)
    except ValidationError as exc:
        errors = {str(error["loc"][0]): str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()}
        return render(request, "error.html", status_code=422, title="Check your inputs", message="Correct the fields below to build your report.",
                      ticker=ticker, target_date=target_date, errors=errors, retry=True, active="workspace")
    try:
        result = await analyze(analysis_request)
    except HTTPException as exc:
        return render(request, "error.html", status_code=exc.status_code, title="Report unavailable", message=exc.detail,
                      ticker=analysis_request.ticker, target_date=analysis_request.target_date, retry=True, active="workspace")
    return render(request, "report.html", title=f"{result.ticker} report", active="workspace", result=result,
                  ticker=result.ticker, target_date=result.target_date, report_html=render_report(result.final_report),
                  return_to="/report?" + urlencode({"ticker": result.ticker, "target_date": result.target_date}))


@router.get("/examples")
async def examples(request: Request):
    return render(request, "examples.html", title="Example reports", active="examples")


@router.get("/examples/{slug}")
async def example(request: Request, slug: str):
    entry = EXAMPLES.get(slug)
    if entry is None:
        return render(request, "error.html", status_code=404, title="Example not found", message="Choose one of the three published fixture reports.")
    source = (ROOT / "reports" / entry["file"]).read_text(encoding="utf-8")
    return render(request, "report.html", title=entry["title"], active="examples", fixture=entry,
                  report_html=render_report(source))


@router.get("/methodology")
async def methodology(request: Request):
    return render(request, "methodology.html", title="Methodology", active="methodology")


def local_return(value: str) -> str:
    decoded = unquote(value)
    if (not decoded.startswith("/") or decoded.startswith("//") or "\\" in decoded
            or "%" in decoded or any(ord(char) < 32 or ord(char) == 127 for char in decoded)):
        return "/"
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc:
        return "/"
    return value


@router.post("/theme")
async def theme(request: Request, theme: str = Form(""), return_to: str = Form("/")):
    if theme not in {"light", "dark", "system"}:
        return render(request, "error.html", status_code=422, title="Choose a theme", message="Select Light, Dark or System using the appearance controls.")
    response = RedirectResponse(local_return(return_to), status_code=303)
    response.set_cookie("theme", theme, max_age=31536000, httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return response
