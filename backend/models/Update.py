from typing import Optional
from pydantic import BaseModel

class LineupUpdate(BaseModel):
    home: str
    away: str
    match_date: str
    home_lineup: list
    away_lineup: list
    source: Optional[str] = "manual"

class InjuryUpdate(BaseModel):
    team: str
    player: str
    status: str  # "out", "doubt", "available"
    return_date: Optional[str] = None
    source: Optional[str] = "manual"

class StatsUpdate(BaseModel):
    match_id: Optional[str] = None
    home: str
    away: str
    match_date: str
    data: dict
