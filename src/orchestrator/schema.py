from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class JobInfo(BaseModel):
    title: str
    company: str
    requirements: dict


class CandidateInfo(BaseModel):
    name: str
    current_position: Optional[str] = None
    years_experience: Optional[int] = None
    key_skills: List[str]


class QA(BaseModel):
    role: Literal["human", "ai"]
    text: str
    audio_path: Optional[str] = None
    ts: float = Field(..., description="Unix timestamp")
