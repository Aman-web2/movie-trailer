"""
youtube.py
YouTube trailer search, video selection, comment retrieval, and API error handling.
Uses YouTube Data API v3.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
BASE_URL        = "https://www.googleapis.com/youtube/v3"
TIMEOUT         = 15  # seconds


def has_credentials() -> bool:
    """Return True if a YouTube API key is configured."""
    return bool(YOUTUBE_API_KEY)


def search_trailers(movie_title: str, max_results: int = 6) -> list | dict:
    """
    Search YouTube for trailers related to *movie_title*.

    Returns a list of video info dicts on success:
        [{ 'video_id', 'title', 'thumbnail', 'channel' }, ...]
    or a dict with an 'error' key on failure.
    """
    if not has_credentials():
        return {"error": "YouTube API key missing. Add YOUTUBE_API_KEY to your .env file."}

    if not movie_title or not movie_title.strip():
        return {"error": "Movie title is required to search for trailers."}

    url    = f"{BASE_URL}/search"
    params = {
        "part":       "snippet",
        "q":          f"{movie_title.strip()} official trailer",
        "type":       "video",
        "maxResults": min(max(1, max_results), 10),
        "key":        YOUTUBE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)

        if resp.status_code == 400:
            return {"error": "Invalid YouTube API request. Check your API key."}
        if resp.status_code == 403:
            data = resp.json()
            reason = _extract_yt_reason(data)
            if "quotaExceeded" in reason:
                return {"error": "YouTube API quota exceeded. Try again tomorrow or use a different key."}
            return {"error": f"YouTube API access denied: {reason}"}
        if resp.status_code == 404:
            return {"error": "No trailers found for this movie."}

        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return {"error": f'No trailers found for "{movie_title}".'}

        trailers = []
        for item in items:
            vid_id = item.get("id", {}).get("videoId")
            snip   = item.get("snippet", {})
            if not vid_id:
                continue
            trailers.append({
                "video_id":  vid_id,
                "title":     snip.get("title", "Unknown"),
                "thumbnail": snip.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "channel":   snip.get("channelTitle", ""),
            })

        return trailers if trailers else {"error": "No valid trailer videos returned."}

    except requests.Timeout:
        return {"error": "YouTube request timed out. Check your internet connection."}
    except requests.HTTPError as exc:
        return {"error": f"YouTube HTTP error {exc.response.status_code}."}
    except requests.RequestException as exc:
        return {"error": f"Network error while contacting YouTube: {exc}"}
    except ValueError:
        return {"error": "YouTube returned an unexpected response format."}


def get_video_comments(video_id: str, max_results: int = 100) -> list | dict:
    """
    Fetch top-level public comments for a YouTube video.

    Returns a list of comment strings on success,
    or a dict with an 'error' key on failure.

    Handles:
    - Disabled comments
    - Private / unavailable videos
    - Quota exceeded
    - Invalid API key
    - Network failures
    """
    if not has_credentials():
        return {"error": "YouTube API key missing. Add YOUTUBE_API_KEY to your .env file."}

    if not video_id or not video_id.strip():
        return {"error": "A valid YouTube video ID is required."}

    url    = f"{BASE_URL}/commentThreads"
    params = {
        "part":       "snippet",
        "videoId":    video_id.strip(),
        "maxResults": min(max(1, max_results), 100),
        "textFormat": "plainText",
        "order":      "relevance",
        "key":        YOUTUBE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)

        if resp.status_code == 400:
            data   = resp.json()
            reason = _extract_yt_reason(data)
            if "commentsDisabled" in reason:
                return {"error": "Comments are disabled for this video. Please select a different trailer."}
            return {"error": f"YouTube API bad request: {reason}"}

        if resp.status_code == 403:
            data   = resp.json()
            reason = _extract_yt_reason(data)
            if "quotaExceeded" in reason:
                return {"error": "YouTube API quota exceeded. Try again tomorrow."}
            if "commentsDisabled" in reason:
                return {"error": "Comments are disabled for this video."}
            return {"error": f"YouTube API access denied: {reason}"}

        if resp.status_code == 404:
            return {"error": "Video not found. It may have been removed."}

        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return {"error": "No comments found for this video."}

        comments = []
        for item in items:
            try:
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                if text and text.strip():
                    comments.append(text.strip())
            except (KeyError, TypeError):
                continue  # skip malformed items

        if not comments:
            return {"error": "Comments were returned but could not be parsed."}

        return comments

    except requests.Timeout:
        return {"error": "YouTube request timed out. Check your internet connection."}
    except requests.HTTPError as exc:
        return {"error": f"YouTube HTTP error {exc.response.status_code}."}
    except requests.RequestException as exc:
        return {"error": f"Network error while contacting YouTube: {exc}"}
    except ValueError:
        return {"error": "YouTube returned an unexpected response format."}


def embed_url(video_id: str) -> str:
    """Return the YouTube embed URL for a video ID."""
    return f"https://www.youtube.com/embed/{video_id}"


def watch_url(video_id: str) -> str:
    """Return the standard YouTube watch URL for a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_yt_reason(data: dict) -> str:
    """Pull the first error reason string from a YouTube API error response."""
    try:
        return data["error"]["errors"][0].get("reason", "")
    except (KeyError, IndexError, TypeError):
        return ""
