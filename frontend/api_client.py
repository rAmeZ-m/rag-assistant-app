import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
TIMEOUT = 180  # seconds; a local model can be slow


class ApiError(Exception):
    """An error message that is safe to show to the user."""


def _base_url() -> str:
    if not API_BASE_URL:
        raise ApiError("API_BASE_URL is not set. Add it to frontend/.env")
    return API_BASE_URL.rstrip("/")


def ask(question: str) -> dict:
    try:
        r = requests.post(
            f"{_base_url()}/query", json={"question": question}, timeout=TIMEOUT
        )
    except requests.ConnectionError:
        raise ApiError("Can't reach the backend. Make sure it is running.")
    except requests.Timeout:
        raise ApiError("The request took too long. Please try again.")

    if r.status_code == 422:
        raise ApiError("Please enter a valid question (3 to 500 characters).")
    if r.status_code == 503:
        raise ApiError("The language model is unavailable. Make sure Ollama is running.")
    if not r.ok:
        raise ApiError(f"Unexpected error from the backend (status {r.status_code}).")
    return r.json()


def health() -> dict | None:
    try:
        r = requests.get(f"{_base_url()}/health", timeout=5)
        return r.json() if r.ok else None
    except (requests.RequestException, ApiError):
        return None