# AI News Tracker

每日自动追踪 AI 领域最新新闻，使用 Claude 进行智能摘要，通过邮件推送中文日报。

## 功能

- 从 7+ 个 RSS 源抓取 AI 新闻（TechCrunch, The Verge, MIT Tech Review 等）
- 使用 Claude API 智能总结，生成中文日报
- 通过 SMTP 发送精美 HTML 邮件
- 支持定时任务，每天自动执行
- 支持 dry-run 模式，先预览再发送

## 快速开始

### 1. 安装依赖

```bash
pip install -r ai_news_tracker/requirements.txt
```

### 2. 配置环境变量

```bash
cp ai_news_tracker/.env.example .env
```

编辑 `.env` 文件，填写：

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API Key（[获取](https://console.anthropic.com/)） |
| `SMTP_HOST` | 邮件服务器（默认 smtp.gmail.com） |
| `SMTP_PORT` | 端口（默认 587） |
| `SMTP_USER` | 发件邮箱 |
| `SMTP_PASSWORD` | 邮箱应用密码 |
| `EMAIL_TO` | 收件邮箱 |

> **Gmail 用户**: 需要开启两步验证，然后生成[应用专用密码](https://myaccount.google.com/apppasswords)

### 3. 运行

```bash
# 预览模式（不发邮件，输出到终端）
python -m ai_news_tracker --dry-run

# 立即运行一次
python -m ai_news_tracker

# 定时模式（每天 08:00 自动发送）
python -m ai_news_tracker --schedule --time 08:00
```

### 4. 设置 Cron 定时任务（推荐）

```bash
# 编辑 crontab
crontab -e

# 添加一行（每天早上 8 点运行）
0 8 * * * cd /path/to/Signal && python -m ai_news_tracker
```

## 项目结构

```
ai_news_tracker/
├── __init__.py        # 包初始化
├── __main__.py        # python -m 入口
├── main.py            # 主程序和调度器
├── config.py          # 配置管理
├── news_fetcher.py    # RSS 新闻抓取
├── summarizer.py      # Claude AI 摘要
├── emailer.py         # 邮件发送
├── requirements.txt   # 依赖
├── .env.example       # 环境变量模板
└── README.md          # 说明文档
```

## 新闻源

| 源 | 类型 |
|---|------|
| TechCrunch AI | 科技新闻 |
| The Verge AI | 科技新闻 |
| MIT Technology Review | 学术/产业 |
| VentureBeat AI | 产业/投资 |
| Ars Technica AI | 深度报道 |
| Google AI Blog | 官方博客 |
| OpenAI Blog | 官方博客 |

可在 `config.py` 的 `RSS_FEEDS` 中自定义添加或删除新闻源。
