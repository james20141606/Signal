"""Fetch AI news from RSS feeds."""

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import feedparser

from .config import RSS_FEEDS, MAX_ARTICLES_PER_FEED, MAX_ARTICLES_TOTAL

logger = logging.getLogger(__name__)


@dataclass
class Article:
    title: str
    link: str
    summary: str
    source: str
    published: datetime


def fetch_news(hours: int = 24) -> list[Article]:
    """Fetch AI news articles from the last `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles: list[Article] = []

    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            count = 0
            for entry in feed.entries:
                if count >= MAX_ARTICLES_PER_FEED:
                    break

                # Parse publication date
                published = _parse_date(entry)
                if published and published < cutoff:
                    continue

                summary = ""
                if hasattr(entry, "summary"):
                    summary = _strip_html(entry.summary)[:500]

                articles.append(Article(
                    title=entry.get("title", "No Title"),
                    link=entry.get("link", ""),
                    summary=summary,
                    source=feed_info["name"],
                    published=published or datetime.now(timezone.utc),
                ))
                count += 1

            logger.info(f"Fetched {count} articles from {feed_info['name']}")
        except Exception as e:
            logger.warning(f"Failed to fetch {feed_info['name']}: {e}")

    # Sort by date, newest first
    articles.sort(key=lambda a: a.published, reverse=True)
    return articles[:MAX_ARTICLES_TOTAL]


def _parse_date(entry) -> datetime | None:
    """Parse the publication date from a feed entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        from time import mktime
        return datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        from time import mktime
        return datetime.fromtimestamp(mktime(entry.updated_parsed), tz=timezone.utc)
    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", clean).strip()
