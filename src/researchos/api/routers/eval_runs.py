"""Evaluation run endpoints.

Read-only. An evaluation costs embedding and generation quota, so triggering
one belongs on the CLI where the cost is visible and where a long run can
report progress. Exposing it over HTTP would invite someone to trigger a run
that exhausts the daily quota from a browser button.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from researchos.evaluation.runner import list_runs, run_results

router = APIRouter(prefix="/api/evals", tags=["evaluations"])


class EvalRunOut(BaseModel):
    id: str
    name: str
    started_at: str
    finished_at: str | None
    summary: dict


@router.get("")
def list_eval_runs() -> dict:
    """Recent evaluation runs, newest first."""
    rows = list_runs()
    return {
        "runs": [
            EvalRunOut(
                id=str(row["id"]),
                name=row["name"],
                started_at=row["started_at"].isoformat(),
                finished_at=row["finished_at"].isoformat() if row.get("finished_at") else None,
                summary=row["summary"],
            ).model_dump()
            for row in rows
        ],
        "total": len(rows),
    }


@router.get("/{run_id}")
def get_eval_run(run_id: uuid.UUID) -> dict:
    """One run's summary plus per-question detail."""
    run = None
    for row in list_runs():
        if str(row["id"]) == str(run_id):
            run = row
            break

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no evaluation run with id {run_id}",
        )

    return {
        "run": {
            "id": str(run["id"]),
            "name": run["name"],
            "started_at": run["started_at"].isoformat(),
            "finished_at": run["finished_at"].isoformat() if run.get("finished_at") else None,
            "summary": run["summary"],
        },
        "results": [
            {
                "question": row["question"],
                "expected_sources": row["expected_sources"],
                "retrieved": row["retrieved"],
                "answer": row["answer"],
                "metrics": row["metrics"],
                "latencies": row["latencies"],
                "error": row["error"],
            }
            for row in run_results(run_id)
        ],
    }
