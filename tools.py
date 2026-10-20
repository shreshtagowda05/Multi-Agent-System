
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

# ---------- Tavily setup ----------
api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Check your .env file."
    )

tavily = TavilyClient(api_key=api_key)


# ---------- Search tool schema ----------
class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The research topic or search query to search for on the web."
    )


# ---------- Web search tool ----------
@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.
    Returns source titles, URLs, and snippets.
    Always provide the required query argument.
    """
    try:
        results = tavily.search(
            query=query,
            max_results=5
        )

        search_results = results.get("results", [])

        if not search_results:
            return f"No search results found for: {query}"

        output = []

        for result in search_results:
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No snippet")

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {content[:300]}\n"
            )

        return "\n---\n".join(output)

    except Exception as e:
        return f"Web search failed: {type(e).__name__}: {e}"


# ---------- URL scraping schema ----------
class ScrapeURLInput(BaseModel):
    url: str = Field(
        ...,
        description="The complete HTTP or HTTPS URL of the webpage to scrape."
    )


# ---------- URL scraping tool ----------
@tool("scrape_url", args_schema=ScrapeURLInput)
def scrape_url(url: str) -> str:
    """Scrape a webpage and return clean text for deeper research.
    Requires a complete HTTP or HTTPS URL.
    """
    try:
        if not url.startswith(("http://", "https://")):
            return "Invalid URL: provide a complete HTTP or HTTPS URL."

        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if not text:
            return f"No readable text found at {url}"

        return text[:5000]

    except requests.RequestException as e:
        return f"Could not scrape URL: {type(e).__name__}: {e}"
    except Exception as e:
        return f"Scraping failed: {type(e).__name__}: {e}"
