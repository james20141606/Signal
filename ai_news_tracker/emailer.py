"""Send news digest via email using SMTP."""

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from jinja2 import Template

from .config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO, EMAIL_FROM_NAME

logger = logging.getLogger(__name__)

EMAIL_TEMPLATE = Template("""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 700px; margin: 0 auto; padding: 20px; color: #333; background: #f5f5f5; }
  .container { background: #fff; border-radius: 12px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  h1 { color: #1a73e8; font-size: 24px; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }
  .date { color: #666; font-size: 14px; margin-bottom: 20px; }
  .content { line-height: 1.8; white-space: pre-wrap; }
  .footer { margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee;
            color: #999; font-size: 12px; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <h1>🤖 AI 新闻日报</h1>
  <div class="date">{{ date }}</div>
  <div class="content">{{ content }}</div>
  <div class="footer">
    由 AI News Tracker 自动生成 | 共追踪 {{ article_count }} 条新闻
  </div>
</div>
</body>
</html>
""")


def send_email(summary: str, article_count: int) -> bool:
    """Send the news digest email."""
    if not all([SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
        logger.error("Email configuration incomplete. Set SMTP_USER, SMTP_PASSWORD, and EMAIL_TO in .env")
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subject = f"AI 新闻日报 - {today}"

    # Build HTML email
    html_content = EMAIL_TEMPLATE.render(
        date=today,
        content=summary,
        article_count=article_count,
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{EMAIL_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = EMAIL_TO

    # Plain text fallback
    msg.attach(MIMEText(summary, "plain", "utf-8"))
    # HTML version
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent to {EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
