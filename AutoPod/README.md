# AutoPod: Stork PDF Digest Podcast

This project pulls PDF attachments from 163 IMAP emails (subject contains `Stork`),
summarizes each PDF with OpenAI, generates Chinese audio (female voice), and publishes
episodes to a private RSS feed stored on Cloudflare R2.

## Architecture
- GitHub Actions schedule (07:00 Asia/Shanghai -> 23:00 UTC)
- 163 IMAP fetch + PDF extraction
- OpenAI summary + TTS
- R2 storage for `mp3` and `rss.xml`

## Quick start
1) Create a Cloudflare R2 bucket and enable a public base URL (custom domain or `r2.dev`).
2) Enable IMAP/SMTP on your 163 mailbox and create an app password.
3) Add GitHub Actions secrets (see below).
4) Push this repo to GitHub and enable Actions.

## GitHub Actions secrets
Required:
- `IMAP_USER`: your 163 email address
- `IMAP_PASSWORD`: 163 app password (not the login password)
- `OPENAI_API_KEY`
- `R2_ENDPOINT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`
- `R2_PUBLIC_BASE_URL`
- `RSS_PRIVATE_TOKEN`: any random string used in the feed URL

Optional:
- `PODCAST_TITLE`
- `PODCAST_DESCRIPTION`
- `PODCAST_LINK`

## Feed URL
If `R2_PUBLIC_BASE_URL` is `https://<account>.r2.dev/<bucket>` and
`RSS_PRIVATE_TOKEN` is `abc123`, then the feed will be:

`https://<account>.r2.dev/<bucket>/autopod/rss_abc123.xml`

## Local run (optional)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
IMAP_USER=... IMAP_PASSWORD=... OPENAI_API_KEY=... R2_ENDPOINT=... \
R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=... \
R2_PUBLIC_BASE_URL=... RSS_PRIVATE_TOKEN=... python -m autopod.main
```

## Notes
- The script stores a small `state.json` in R2 to avoid reprocessing PDFs.
- Each paper is 1-2 minutes; episodes are grouped up to 30 minutes each.
