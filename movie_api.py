"""
movie_api.py
TMDB authentication, movie search, movie details, poster URLs, and error handling.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY     = os.getenv("TMDB_API_KEY", "").strip()
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()

BASE_URL         = "https://api.themoviedb.org/3"
POSTER_BASE      = "https://image.tmdb.org/t/p/w500"
POSTER_SMALL     = "https://image.tmdb.org/t/p/w185"

TIMEOUT = 15  # seconds


def _headers() -> dict:
    """Build request headers, preferring Bearer token when available."""
    h = {"accept": "application/json"}
    if TMDB_BEARER_TOKEN:
        h["Authorization"] = f"Bearer {TMDB_BEARER_TOKEN}"
    return h


def _auth_params() -> dict:
    """Return api_key param when no Bearer token is configured."""
    if not TMDB_BEARER_TOKEN and TMDB_API_KEY:
        return {"api_key": TMDB_API_KEY}
    return {}


def has_credentials() -> bool:
    """Return True if at least one TMDB credential is configured."""
    return bool(TMDB_API_KEY or TMDB_BEARER_TOKEN)


def search_movies(query: str) -> list | dict:
    """
    Search TMDB for movies matching *query*.

    Returns a list of movie result dicts on success,
    or a dict with an 'error' key on failure.
    """
    if not has_credentials():
        return {"error": "TMDB API credentials missing. Add TMDB_API_KEY to your .env file."}

    if not query or not query.strip():
        return {"error": "Please enter a movie name to search."}

    url    = f"{BASE_URL}/search/movie"
    params = {**_auth_params(), "query": query.strip(), "language": "en-US", "page": 1}

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)

        if resp.status_code == 401:
            return {"error": "TMDB authentication failed. Check your API key in .env."}
        if resp.status_code == 404:
            return {"error": "No movies found for that search."}
        if resp.status_code == 429:
            return {"error": "TMDB rate limit reached. Please wait a moment and try again."}

        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            return {"error": f'No movies found for "{query}". Try a different title.'}

        return results

    except requests.Timeout:
        return {"error": "TMDB request timed out. Check your internet connection."}
    except requests.HTTPError as exc:
        return {"error": f"TMDB HTTP error {exc.response.status_code}."}
    except requests.RequestException as exc:
        return {"error": f"Network error while contacting TMDB: {exc}"}
    except ValueError:
        return {"error": "TMDB returned an unexpected response format."}


def get_movie_details(movie_id: int) -> dict:
    """
    Fetch full details for a single movie by its TMDB ID.

    Returns a movie detail dict or a dict with an 'error' key.
    """
    if not has_credentials():
        return {"error": "TMDB API credentials missing."}

    url    = f"{BASE_URL}/movie/{movie_id}"
    params = {**_auth_params(), "language": "en-US"}

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)

        if resp.status_code == 401:
            return {"error": "TMDB authentication failed."}
        if resp.status_code == 404:
            return {"error": "Movie not found on TMDB."}

        resp.raise_for_status()
        return resp.json()

    except requests.Timeout:
        return {"error": "TMDB request timed out."}
    except requests.RequestException as exc:
        return {"error": f"Network error: {exc}"}
    except ValueError:
        return {"error": "TMDB returned an unexpected response format."}


def poster_url(path: str, small: bool = False) -> str | None:
    """
    Build a full poster URL from a TMDB poster_path fragment.
    Returns None if path is empty.
    """
    if not path:
        return None
    base = POSTER_SMALL if small else POSTER_BASE
    return f"{base}{path}"


def format_movie(raw: dict) -> dict:
    """
    Normalise a raw TMDB movie result into a cleaner dict
    with the fields the UI needs.
    """
    return {
        "id":          raw.get("id"),
        "title":       raw.get("title", "Unknown Title"),
        "overview":    raw.get("overview", "No overview available."),
        "release":     raw.get("release_date", "")[:4] or "N/A",
        "rating":      raw.get("vote_average", 0.0),
        "votes":       raw.get("vote_count", 0),
        "poster":      poster_url(raw.get("poster_path")),
        "poster_sm":   poster_url(raw.get("poster_path"), small=True),
        "popularity":  raw.get("popularity", 0),
        "genres":      [g["name"] for g in raw.get("genres", [])],
    }
