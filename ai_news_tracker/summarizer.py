"""Use Claude API to summarize AI news articles."""

import logging

import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .news_fetcher import Article

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的 AI 行业新闻编辑。你的任务是将收集到的 AI 新闻整理成一份精炼的中文日报。

要求：
1. 用中文撰写摘要
2. 按重要性排序，最重要的新闻放在前面
3. 每条新闻用 1-2 句话概括核心内容
4. 标注新闻来源
5. 如果有多条新闻讨论同一话题，合并为一条
6. 在末尾给出一段"今日总结"，概括今天 AI 领域的整体动态"""


def summarize_news(articles: list[Article]) -> str:
    """Use Claude to generate a Chinese summary of the news articles."""
    if not articles:
        return "今日暂无 AI 新闻更新。"

    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY set, returning raw article list")
        return _fallback_summary(articles)

    # Build article text for Claude
    articles_text = "\n\n".join(
        f"【{a.source}】{a.title}\n链接: {a.link}\n摘要: {a.summary}"
        for a in articles
    )

    prompt = f"以下是今天收集到的 {len(articles)} 条 AI 相关新闻，请整理成一份中文日报：\n\n{articles_text}"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return _fallback_summary(articles)


def _fallback_summary(articles: list[Article]) -> str:
    """Generate a simple summary without AI when API is unavailable."""
    lines = [f"📰 AI 新闻日报 ({len(articles)} 条)\n"]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. 【{a.source}】{a.title}")
        if a.summary:
            lines.append(f"   {a.summary[:200]}")
        lines.append(f"   🔗 {a.link}\n")
    return "\n".join(lines)
