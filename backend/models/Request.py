from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    home: str
    away: str
    match_date: Optional[str] = None
    match_time: Optional[str] = None
    odds_h: Optional[float] = None
    odds_d: Optional[float] = None
    odds_a: Optional[float] = None