from pydantic import BaseModel, Field
from typing import List, Optional

class ScoreBreakdown(BaseModel):
    skills_match: int = Field(..., ge=0, le=100)
    impact_and_metrics: int = Field(..., ge=0, le=100)
    formatting_and_sections: int = Field(..., ge=0, le=100)

class ExtractedProfile(BaseModel):
    detected_role: str
    matched_skills: List[str]
    missing_skills: List[str]

class SuggestedChange(BaseModel):
    section: str
    severity: str  # "high", "medium", "low"
    issue: str
    original_text: Optional[str] = None
    recommendation: str

class CVAnalysisResponse(BaseModel):
    filename: str
    overall_score: int = Field(..., ge=0, le=100)
    score_breakdown: ScoreBreakdown
    extracted_profile: ExtractedProfile
    suggested_changes: List[SuggestedChange]