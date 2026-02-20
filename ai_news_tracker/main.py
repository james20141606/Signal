#!/usr/bin/env python3
"""AI News Tracker - Daily AI news digest with email delivery.

Usage:
    python -m ai_news_tracker              # Run once: fetch, summarize, send email
    python -m ai_news_tracker --schedule   # Run as a scheduler (sends daily at 08:00)
    python -m ai_news_tracker --dry-run    # Fetch & summarize, print to console (no email)
"""

import argparse
import logging
import sys
import time

import schedule

from .news_fetcher import fetch_news
from .summarizer import summarize_news
from .emailer import send_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_once(dry_run: bool = False) -> None:
    """Fetch news, summarize, and send email."""
    logger.info("Fetching AI news...")
    articles = fetch_news(hours=24)
    logger.info(f"Fetched {len(articles)} articles")

    if not articles:
        logger.info("No new articles found, skipping.")
        return

    logger.info("Generating summary with Claude...")
    summary = summarize_news(articles)

    if dry_run:
        print("\n" + "=" * 60)
        print("AI NEWS DIGEST (dry run)")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        return

    logger.info("Sending email...")
    success = send_email(summary, len(articles))
    if success:
        logger.info("Done! Email sent successfully.")
    else:
        logger.error("Failed to send email.")
        sys.exit(1)


def run_scheduler(send_time: str = "08:00") -> None:
    """Run the tracker on a daily schedule."""
    logger.info(f"Scheduler started. Will send daily at {send_time}")
    schedule.every().day.at(send_time).do(run_once)

    # Also run once immediately
    run_once()

    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI News Tracker")
    parser.add_argument("--schedule", action="store_true", help="Run as daily scheduler")
    parser.add_argument("--time", default="08:00", help="Daily send time (HH:MM, default: 08:00)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch & summarize, print to console")
    args = parser.parse_args()

    if args.schedule:
        run_scheduler(args.time)
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
