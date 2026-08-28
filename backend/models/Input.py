from typing import Optional
from pydantic import BaseModel

class OddsInput(BaseModel):
    home: str
    away: str
    odds: dict
    min_edge: Optional[float] = 0.07
    bankroll: Optional[float] = 300.0
    match_date: Optional[str] = None
