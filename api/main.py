"""FastAPI service exposing the arbitration pipeline.

Run with:  uvicorn api.main:app --reload
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from arbitration.config import load_settings
from arbitration.graph import run_arbitration
from arbitration.models import ArbitrationRecord
from arbitration.storage import count_arbitrations, get_arbitration, list_arbitrations, save_arbitration

app = FastAPI(
    title="Quorum",
    description=(
        "Routes an LLM-generated output to three independent critic agents "
        "(accuracy, logic, completeness), detects where they disagree, and "
        "synthesizes their critiques into a single confidence-scored verdict."
    ),
    version="0.1.0",
)


class ArbitrateRequest(BaseModel):
    output: str = Field(..., description="The LLM-generated output to evaluate.")
    prompt: str | None = Field(None, description="The original prompt/question the output was responding to.")


class BatchArbitrateRequest(BaseModel):
    items: list[ArbitrateRequest] = Field(..., min_length=1, max_length=100)


class BatchArbitrateResponse(BaseModel):
    results: list[ArbitrationRecord]


@app.get("/v1/health")
def health() -> dict:
    settings = load_settings()
    return {"status": "ok", "provider_mode": settings.provider_mode}


@app.post("/v1/arbitrate", response_model=ArbitrationRecord)
def arbitrate(request: ArbitrateRequest) -> ArbitrationRecord:
    if not request.output.strip():
        raise HTTPException(status_code=422, detail="`output` must not be empty")
    settings = load_settings()
    record = run_arbitration(request.output, request.prompt)
    save_arbitration(settings.db_path, record)
    return record


@app.post("/v1/arbitrate/batch", response_model=BatchArbitrateResponse)
def arbitrate_batch(request: BatchArbitrateRequest) -> BatchArbitrateResponse:
    settings = load_settings()
    results: list[ArbitrationRecord] = []
    for item in request.items:
        if not item.output.strip():
            continue
        record = run_arbitration(item.output, item.prompt)
        save_arbitration(settings.db_path, record)
        results.append(record)
    return BatchArbitrateResponse(results=results)


@app.get("/v1/arbitrations", response_model=list[ArbitrationRecord])
def list_recent_arbitrations(limit: int = 50, offset: int = 0) -> list[ArbitrationRecord]:
    settings = load_settings()
    return list_arbitrations(settings.db_path, limit=limit, offset=offset)


@app.get("/v1/arbitrations/count")
def arbitrations_count() -> dict:
    settings = load_settings()
    return {"count": count_arbitrations(settings.db_path)}


# NOTE: this dynamic route must be registered after the static /v1/arbitrations
# and /v1/arbitrations/count routes above, or it would shadow them.
@app.get("/v1/arbitrations/{arbitration_id}", response_model=ArbitrationRecord)
def get_arbitration_by_id(arbitration_id: str) -> ArbitrationRecord:
    settings = load_settings()
    record = get_arbitration(settings.db_path, arbitration_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No arbitration found with id={arbitration_id!r}")
    return record
