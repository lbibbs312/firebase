# google_cse.py
import os
import requests

def google_cse_search(query: str, num_results: int = 3) -> str:
    """
    Calls Google CSE with the given query, returns a textual summary of results.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
    if not api_key or not cse_id:
        return "Missing GOOGLE_API_KEY or GOOGLE_CUSTOM_SEARCH_ENGINE_ID in .env."

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,  # how many results to return
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return f"Error from Google CSE: {resp.status_code} {resp.text}"

    data = resp.json()
    items = data.get("items")
    if not items:
        return "No search results found."

    # Build a human-readable string of top results
    results_str = []
    for i, item in enumerate(items, start=1):
        title = item.get("title", "No title")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        results_str.append(f"\nResult {i}:\nTitle: {title}\nURL: {link}\nSnippet: {snippet}\n")
    return "\n".join(results_str)
