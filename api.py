import time

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import ask

app = FastAPI(title="SafeSite Assistant")


class Question(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: int
    excerpt: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


@app.post("/ask", response_model=Answer)
def ask_endpoint(body: Question) -> Answer:
    start = time.perf_counter()
    try:
        answer, docs = ask(body.question)
    except openai.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="The language model is rate-limited. Please retry in a moment.",
        )
    except openai.APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The language model timed out. Please retry.",
        )
    except openai.OpenAIError:
        raise HTTPException(
            status_code=502,
            detail="The language model service failed.",
        )
    return Answer(
        answer=answer,
        sources=[
            Source(
                file=d.metadata["source"],
                page=d.metadata["page"],
                excerpt=d.page_content[:200],
            )
            for d in docs
        ],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )