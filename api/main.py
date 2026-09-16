from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.frontend import ROOT, router as frontend_router
from api.routes import router

app = FastAPI(
    title="Market-Agent-Ops API",
    description="API for the Market-Agent-Ops pipeline.",
    version="1.0.0"
)


app.include_router(router)
app.include_router(frontend_router)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
