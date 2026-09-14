from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Market-Agent-Ops API",
    description="API for the Market-Agent-Ops pipeline.",
    version="1.0.0"
)


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)