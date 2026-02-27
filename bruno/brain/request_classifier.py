"""
Classify whether a transcript is likely a real sentence or request worth answering,
using an LLM instead of keyword patterns.
"""
from bruno.brain.ollama_client import ollama_chat

CLASSIFIER_MODEL = "qwen2.5:7b"

SYSTEM = """You judge if an utterance is a real sentence or request that deserves a helpful reply.
Ignore filler, single words, background noise, or unclear fragments.
Reply with exactly one word: YES or NO."""


def is_likely_real_request(transcript: str) -> bool:
    """
    Return True if the transcript is likely a real sentence or request worth answering.
    Uses the LLM to analyze likelihood instead of keyword matching.
    """
    text = (transcript or "").strip()
    if len(text) < 3:
        return False
    # Cap length so we don't send huge noise to the model
    text = text[:400]
    prompt = f"{SYSTEM}\n\nUtterance: \"{text}\""
    try:
        out = ollama_chat(CLASSIFIER_MODEL, prompt).strip().upper()
        # Treat as YES if the model says yes in the first bit of its reply
        return "YES" in (out[:20] if len(out) > 20 else out)
    except Exception:
        return False
