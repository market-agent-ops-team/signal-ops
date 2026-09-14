from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime

from agents.graph import app as graph_app

router = APIRouter()



class AnalyzeRequest(BaseModel):
    ticker: str
    target_date: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")
        return v.strip()

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        if not v:
            raise ValueError("Target date is required")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        return v

class AnalyzeResponse(BaseModel):
    ticker: str
    target_date: str
    ml_trend: str
    ml_confidence: float
    news_sentiment: str
    divergence_flag: bool
    analyst_reasoning: str
    final_report: str



@router.get("/health")
async def health_check():
   
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
   
    try:
        initial_state = {
            "ticker": request.ticker,
            "target_date": request.target_date
        }
        
       
        final_state = graph_app.invoke(initial_state)
        
       
        return AnalyzeResponse(
            ticker=final_state.get("ticker", request.ticker),
            target_date=final_state.get("target_date", request.target_date),
            ml_trend=final_state.get("ml_trend", ""),
            ml_confidence=final_state.get("ml_confidence", 0.0),
            news_sentiment=final_state.get("news_sentiments", ""),
            divergence_flag=final_state.get("divergence_flag", False),
            analyst_reasoning=final_state.get("analyst_reasoning", ""),
            final_report=final_state.get("final_report", "")
        )
        
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")