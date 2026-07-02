"""
Request/response schema for POST /chat.

This schema supports stateless conversation history, explicit
conversation state, and grounded SHL recommendations.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ConversationState(BaseModel):
    role: Optional[str] = None
    industry: Optional[str] = None
    experience: Optional[str] = None
    job_level: Optional[str] = None
    assessment_types: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    personality_requirement: Optional[str] = None
    cognitive_requirement: Optional[str] = None
    language: Optional[str] = None
    accent: Optional[str] = None
    duration_minutes: Optional[int] = None
    remote_on_site: Optional[str] = None
    candidate_volume: Optional[str] = None
    purpose: Optional[str] = None
    comparison_request: bool = False
    comparison_targets: List[str] = Field(default_factory=list)
    explicit_remove: List[str] = Field(default_factory=list)
    explicit_final_list: List[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_prompt: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    entity_id: str
    name: str
    url: HttpUrl
    test_type: str
    keys: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    reason: str
    confidence: Literal["high", "medium", "low"]
    matched_constraints: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)

    @field_validator("matched_constraints", "matched_skills", mode="before")
    @classmethod
    def _default_list(cls, v):
        return v or []


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    comparison_table: Optional[str] = None
    state: Optional[ConversationState] = None
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _cap_size(cls, v):
        if v is not None and not (1 <= len(v) <= 10):
            raise ValueError("recommendations must contain between 1 and 10 items")
        return v


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
