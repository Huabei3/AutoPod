import hashlib
import imaplib
import io
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
import email
from email.header import decode_header
from email.message import Message
from email.utils import format_datetime, parsedate_to_datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import boto3
import pdfplumber
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from mutagen.mp3 import MP3
from openai import OpenAI


@dataclass
class Config:
    imap_host: str
    imap_port: int
    imap_user: str
    imap_password: str
    imap_folder: str
    imap_subject_keyword: str
    imap_since_days: int
    openai_api_key: str
    openai_model: str
    openai_tts_model: str
    openai_tts_voice: str
    r2_endpoint: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    r2_public_base_url: str
    rss_private_token: Optional[str]
    podcast_title: str
    podcast_description: str
    podcast_language: str
    podcast_link: str
    podcast_prefix: str
    max_episode_seconds: int
    timezone: str
    max_pdf_chars: int


@dataclass
class PdfItem:
    pdf_hash: str
    subject: str
    filename: str
    message_id: str
    received_at: datetime
    pdf_bytes: bytes


@dataclass
class PaperSummary:
    title: str
    script: str
    audio_bytes: bytes
    duration_sec: float
    pdf_hash: str


@dataclass
class Episode:
    title: str
    description: str
    audio_bytes: bytes
    duration_sec: float
    item_titles: List[str]


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def load_config() -> Config:
    return Config(
        imap_host=get_env("IMAP_HOST", "imap.163.com"),
        imap_port=get_int_env("IMAP_PORT", 993),
        imap_user=get_env("IMAP_USER", required=True),
        imap_password=get_env("IMAP_PASSWORD", required=True),
        imap_folder=get_env("IMAP_FOLDER", "INBOX"),
        imap_subject_keyword=get_env("IMAP_SUBJECT_KEYWORD", "Stork"),
        imap_since_days=get_int_env("IMAP_SINCE_DAYS", 7),
        openai_api_key=get_env("OPENAI_API_KEY", required=True),
        openai_model=get_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_tts_model=get_env("OPENAI_TTS_MODEL", "tts-1"),
        openai_tts_voice=get_env("OPENAI_TTS_VOICE", "shimmer"),
        r2_endpoint=get_env("R2_ENDPOINT", required=True),
        r2_access_key_id=get_env("R2_ACCESS_KEY_ID", required=True),
        r2_secret_access_key=get_env("R2_SECRET_ACCESS_KEY", required=True),
        r2_bucket=get_env("R2_BUCKET", required=True),
        r2_public_base_url=get_env("R2_PUBLIC_BASE_URL", required=True),
        rss_private_token=get_env("RSS_PRIVATE_TOKEN", "") or None,
        podcast_title=get_env("PODCAST_TITLE", "Stork Daily Digest"),
        podcast_description=get_env(
            "PODCAST_DESCRIPTION", "Daily digest of Stork papers"
        ),
        podcast_language=get_env("PODCAST_LANGUAGE", "zh-cn"),
        podcast_link=get_env("PODCAST_LINK", "https://example.com"),
        podcast_prefix=get_env("PODCAST_PREFIX", "autopod").strip("/"),
        max_episode_seconds=get_int_env("MAX_EPISODE_SECONDS", 1800),
        timezone=get_env("TIMEZONE", "Asia/Shanghai"),
        max_pdf_chars=get_int_env("MAX_PDF_CHARS", 20000),
    )


def decode_header_value(value: str) -> str:
    parts = decode_header(value)
    decoded = ""
    for text, charset in parts:
        if isinstance(text, bytes):
            try:
                decoded += text.decode(charset or "utf-8", errors="replace")
            except LookupError:
                decoded += text.decode("utf-8", errors="replace")
        else:
            decoded += text
    return decoded.strip()


def extract_pdf_text(pdf_bytes: bytes, max_chars: int) -> str:
    chunks: List[str] = []
    total = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
                total += len(text)
            if total >= max_chars:
                break
    merged = " ".join(chunks)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged[:max_chars]


def search_imap_messages(
    imap: imaplib.IMAP4_SSL, keyword: str, since_days: int
) -> List[bytes]:
    criteria: List[str] = []
    if since_days > 0:
        since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime(
            "%d-%b-%Y"
        )
        criteria.extend(["SINCE", since_date])
    criteria.extend(["SUBJECT", f'"{keyword}"'])
    status, data = imap.search(None, *criteria)
    if status != "OK":
        return []
    if not data or not data[0]:
        return []
    return data[0].split()


def collect_pdf_items(
    cfg: Config, processed_hashes: Iterable[str]
) -> List[PdfItem]:
    processed = set(processed_hashes)
    imap = imaplib.IMAP4_SSL(cfg.imap_host, cfg.imap_port)
    imap.login(cfg.imap_user, cfg.imap_password)
    imap.select(cfg.imap_folder)

    msg_ids = search_imap_messages(imap, cfg.imap_subject_keyword, cfg.imap_since_days)
    items: List[PdfItem] = []
    for msg_id in msg_ids:
        status, msg_data = imap.fetch(msg_id, "(RFC822)")
        if status != "OK":
            continue
        raw = None
        for part in msg_data:
            if isinstance(part, tuple):
                raw = part[1]
                break
        if not raw:
            continue
        message: Message = email.message_from_bytes(raw, policy=policy.default)
        subject = decode_header_value(message.get("Subject", ""))
        if cfg.imap_subject_keyword.lower() not in subject.lower():
            continue
        msg_id_header = (message.get("Message-ID") or "").strip()
        date_header = message.get("Date")
        received_at = (
            parsedate_to_datetime(date_header)
            if date_header
            else datetime.now(timezone.utc)
        )
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = part.get_content_disposition()
            if disposition not in ("attachment", "inline"):
                continue
            filename = decode_header_value(part.get_filename() or "attachment.pdf")
            if not filename.lower().endswith(".pdf"):
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            pdf_hash = hashlib.sha256(payload).hexdigest()
            if pdf_hash in processed:
                continue
            items.append(
                PdfItem(
                    pdf_hash=pdf_hash,
                    subject=subject,
                    filename=filename,
                    message_id=msg_id_header,
                    received_at=received_at,
                    pdf_bytes=payload,
                )
            )
    imap.logout()
    return items


def summarize_pdf(
    client: OpenAI, cfg: Config, pdf_item: PdfItem, text: str
) -> dict:
    prompt = (
        "You are a scientific podcast script editor. "
        "Write a concise Chinese podcast script for 1-2 minutes. "
        "Keep 250-450 Chinese characters, no markdown, no bullet symbols. "
        "Include key findings, methods, and practical takeaway. "
        "Return JSON only: {\"title\": \"...\", \"script\": \"...\"}."
    )
    user_content = (
        f"Email subject: {pdf_item.subject}\n"
        f"Filename: {pdf_item.filename}\n"
        "Extracted text (may be incomplete):\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"title": pdf_item.subject[:60], "script": content.strip()}
    title = (data.get("title") or pdf_item.subject or "Paper Digest").strip()
    script = (data.get("script") or "").strip()
    return {"title": title, "script": script}


def synthesize_audio(client: OpenAI, cfg: Config, script: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio = client.audio.speech.create(
            model=cfg.openai_tts_model,
            voice=cfg.openai_tts_voice,
            input=script,
            response_format="mp3",
        )
        audio.stream_to_file(tmp_path)
        with open(tmp_path, "rb") as handle:
            return handle.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def mp3_duration_seconds(audio_bytes: bytes) -> float:
    audio = MP3(io.BytesIO(audio_bytes))
    return float(audio.info.length)


def build_episodes(papers: List[PaperSummary], max_seconds: int) -> List[Episode]:
    episodes: List[Episode] = []
    current: List[PaperSummary] = []
    total = 0.0
    for paper in papers:
        if current and total + paper.duration_sec > max_seconds:
            episodes.append(create_episode(current))
            current = []
            total = 0.0
        current.append(paper)
        total += paper.duration_sec
    if current:
        episodes.append(create_episode(current))
    return episodes


def create_episode(papers: List[PaperSummary]) -> Episode:
    combined_audio = b"".join(p.audio_bytes for p in papers)
    titles = [p.title for p in papers]
    description = "Papers: " + "; ".join(titles)
    duration = sum(p.duration_sec for p in papers)
    return Episode(
        title="",
        description=description,
        audio_bytes=combined_audio,
        duration_sec=duration,
        item_titles=titles,
    )


def s3_client(cfg: Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.r2_endpoint,
        aws_access_key_id=cfg.r2_access_key_id,
        aws_secret_access_key=cfg.r2_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
    )


def s3_key(cfg: Config, name: str) -> str:
    prefix = cfg.podcast_prefix.strip("/")
    return f"{prefix}/{name}" if prefix else name


def state_key(cfg: Config) -> str:
    suffix = f"state_{cfg.rss_private_token}.json" if cfg.rss_private_token else "state.json"
    return s3_key(cfg, suffix)


def rss_key(cfg: Config) -> str:
    suffix = f"rss_{cfg.rss_private_token}.xml" if cfg.rss_private_token else "rss.xml"
    return s3_key(cfg, suffix)


def load_state(client, cfg: Config) -> dict:
    key = state_key(cfg)
    try:
        obj = client.get_object(Bucket=cfg.r2_bucket, Key=key)
        data = obj["Body"].read()
        return json.loads(data.decode("utf-8"))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return {"processed_hashes": [], "processed_message_ids": []}
        raise


def save_state(client, cfg: Config, state: dict) -> None:
    key = state_key(cfg)
    body = json.dumps(state, ensure_ascii=True, indent=2).encode("utf-8")
    client.put_object(
        Bucket=cfg.r2_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def load_rss_root(client, cfg: Config) -> ET.Element:
    key = rss_key(cfg)
    try:
        obj = client.get_object(Bucket=cfg.r2_bucket, Key=key)
        data = obj["Body"].read()
        return ET.fromstring(data)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            root = ET.Element("rss", version="2.0")
            root.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
            channel = ET.SubElement(root, "channel")
            ET.SubElement(channel, "title").text = cfg.podcast_title
            ET.SubElement(channel, "link").text = cfg.podcast_link
            ET.SubElement(channel, "description").text = cfg.podcast_description
            ET.SubElement(channel, "language").text = cfg.podcast_language
            ET.SubElement(channel, "lastBuildDate").text = format_datetime(
                datetime.now(timezone.utc)
            )
            return root
        raise


def get_channel(root: ET.Element) -> ET.Element:
    channel = root.find("channel")
    if channel is None:
        channel = ET.SubElement(root, "channel")
    return channel


def existing_guids(channel: ET.Element) -> set:
    return {item.findtext("guid", default="") for item in channel.findall("item")}


def insert_item(channel: ET.Element, item: ET.Element) -> None:
    for index, child in enumerate(list(channel)):
        if child.tag == "item":
            channel.insert(index, item)
            return
    channel.append(item)


def update_last_build(channel: ET.Element) -> None:
    last = channel.find("lastBuildDate")
    if last is None:
        last = ET.SubElement(channel, "lastBuildDate")
    last.text = format_datetime(datetime.now(timezone.utc))


def write_rss(client, cfg: Config, root: ET.Element) -> None:
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    client.put_object(
        Bucket=cfg.r2_bucket,
        Key=rss_key(cfg),
        Body=xml,
        ContentType="application/rss+xml; charset=utf-8",
    )


def public_url(cfg: Config, key: str) -> str:
    return f"{cfg.r2_public_base_url.rstrip('/')}/{key}"


def main() -> None:
    cfg = load_config()
    tz = ZoneInfo(cfg.timezone)
    now_local = datetime.now(tz)
    run_date = now_local.strftime("%Y%m%d")

    storage = s3_client(cfg)
    state = load_state(storage, cfg)
    processed_hashes = set(state.get("processed_hashes", []))

    pdf_items = collect_pdf_items(cfg, processed_hashes)
    if not pdf_items:
        print("No new PDFs found.")
        return

    pdf_items.sort(key=lambda item: item.received_at)
    client = OpenAI(api_key=cfg.openai_api_key)

    summaries: List[PaperSummary] = []
    for item in pdf_items:
        text = extract_pdf_text(item.pdf_bytes, cfg.max_pdf_chars)
        summary = summarize_pdf(client, cfg, item, text)
        script = summary["script"]
        if not script:
            continue
        audio_bytes = synthesize_audio(client, cfg, script)
        duration = mp3_duration_seconds(audio_bytes)
        summaries.append(
            PaperSummary(
                title=summary["title"],
                script=script,
                audio_bytes=audio_bytes,
                duration_sec=duration,
                pdf_hash=item.pdf_hash,
            )
        )
        processed_hashes.add(item.pdf_hash)

    if not summaries:
        print("No summaries generated.")
        return

    episodes = build_episodes(summaries, cfg.max_episode_seconds)
    rss_root = load_rss_root(storage, cfg)
    channel = get_channel(rss_root)
    guid_set = existing_guids(channel)

    for index, episode in enumerate(episodes, start=1):
        episode.title = f"{cfg.podcast_title} {run_date} Part {index}"
        audio_key = s3_key(cfg, f"audio/{run_date}/episode_{index:02d}.mp3")
        storage.put_object(
            Bucket=cfg.r2_bucket,
            Key=audio_key,
            Body=episode.audio_bytes,
            ContentType="audio/mpeg",
        )
        audio_url = public_url(cfg, audio_key)
        if audio_url in guid_set:
            continue
        item = ET.Element("item")
        ET.SubElement(item, "title").text = episode.title
        ET.SubElement(item, "description").text = episode.description
        ET.SubElement(item, "pubDate").text = format_datetime(
            datetime.now(timezone.utc)
        )
        guid = ET.SubElement(item, "guid")
        guid.set("isPermaLink", "false")
        guid.text = audio_url
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", audio_url)
        enclosure.set("length", str(len(episode.audio_bytes)))
        enclosure.set("type", "audio/mpeg")
        insert_item(channel, item)
        guid_set.add(audio_url)

    update_last_build(channel)
    write_rss(storage, cfg, rss_root)

    state["processed_hashes"] = sorted(processed_hashes | {s.pdf_hash for s in summaries})
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(storage, cfg, state)
    print(f"Published {len(episodes)} episode(s).")


if __name__ == "__main__":
    main()
