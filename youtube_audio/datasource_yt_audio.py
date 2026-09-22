import yt_dlp
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


import os, re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import hashlib
import requests
from requests.adapters import HTTPAdapter, Retry


import hashlib
import json

from ingestion import DataSourceBase
from ingestion.core.types import IngestionTask
from ingestion.core.logs import make_log_entry

from file_ops import FileOps, jsonable

home_dir = '/Users/pense/'
path_home_dir = Path(home_dir)
fs = FileOps(path_home_dir)


def sanitize_filename(name: str) -> str:
    """Make a safe deterministic filename."""
    return re.sub(r"[^\w\-]+", "_", name).strip("_")



def download_stream(
    url: str,
    dest_path: str,
    chunk_size: int = 1024 * 1024,
    expected_sha256: str | None = None,
    max_retries: int = 5,
    timeout: int = 10,
):
    """Deterministic streaming downloader using requests."""
    if os.path.exists(dest_path):
        return dest_path

    tmp_path = dest_path + ".tmp"

    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        hasher = hashlib.sha256()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    hasher.update(chunk)

        if expected_sha256:
            digest = hasher.hexdigest()
            if digest != expected_sha256:
                os.remove(tmp_path)
                raise ValueError(
                    f"SHA-256 mismatch: expected {expected_sha256}, got {digest}"
                )

    os.replace(tmp_path, dest_path)
    return dest_path



def download_mp3(url: str, out_dir: str, title: str) -> Path:
    """
    Unified downloader:
    - If URL is YouTube → use yt-dlp
    - Else → stream the media file using requests
    - Then convert to MP3 using ffmpeg (48kbps, mono, 22kHz)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(title)
    mp3_path = out_dir / f"{safe_title}.mp3"

    if mp3_path.exists():
        return mp3_path

    # Detect YouTube URLs
    hostname = urlparse(url).hostname or ""
    is_youtube = "youtube" in hostname or "youtu.be" in hostname

    if is_youtube:
        # Use yt-dlp for YouTube sources
        temp_path = out_dir / f"{safe_title}.%(ext)s"

        subprocess.run([
            "yt-dlp",
            "--cookies-from-browser", "chrome",
            "--user-agent", (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "--force-ipv4",
            "-f", "bestaudio/best",
            "-o", str(temp_path),
            url
        ], check=True)


        downloaded = next(out_dir.glob(f"{safe_title}.*"))

    else:
        # Use streaming downloader for direct media URLs
        # Save original file before ffmpeg conversion
        original_path = out_dir / f"{safe_title}_orig"
        download_stream(url, str(original_path))
        downloaded = original_path

    # Convert to MP3 (compressed)
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(downloaded),
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "48k",
        "-ar", "22050",
        "-ac", "1",
        str(mp3_path)
    ], check=True)

    downloaded.unlink()
    return mp3_path


class YouTubeAudioAdapter:
    """
    Adapter for YouTube audio ingestion:
      - parse_video(url)
      - fetch_channel_index(url)
      - download_audio(url, out_dir, title)
      - normalize(video_record, audio_path)
    """

    # ------------------------------------------------------------
    # INTERNAL: yt-dlp metadata extractor
    # ------------------------------------------------------------
    def _extract(self, url: str) -> Dict[str, Any]:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)

    # ------------------------------------------------------------
    # PICK BEST THUMBNAIL
    # ------------------------------------------------------------
    def _best_thumbnail(self, thumbnails: List[Dict[str, Any]]) -> str | None:
        if not thumbnails:
            return None
        sorted_thumbs = sorted(thumbnails, key=lambda t: t.get("preference", 0))
        return sorted_thumbs[-1]["url"]

    # ------------------------------------------------------------
    # SINGLE VIDEO METADATA
    # ------------------------------------------------------------
    def parse_video(self, url: str) -> Dict[str, Any]:
        info = self._extract(url)

        raw_date = info.get("upload_date")
        publish_date = None
        if raw_date:
            publish_date = datetime.strptime(raw_date, "%Y%m%d").date().isoformat()

        return {
            "video_id": info.get("id"),
            "title": info.get("title"),
            "description": info.get("description"),
            "channel_id": info.get("channel_id"),
            "channel_url": info.get("channel_url"),
            "channel_name": info.get("channel"),
            "publish_date": publish_date,
            "duration": info.get("duration"),
            "thumbnail": self._best_thumbnail(info.get("thumbnails", [])),
            "original_url": url,
        }

    # ------------------------------------------------------------
    # FETCH TAB (videos / streams / live)
    # ------------------------------------------------------------
    def _fetch_tab(self, base: str, tab: str):
        url = base.rstrip("/") + f"/{tab}"
        opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                data = ydl.extract_info(url, download=False)
                return data.get("entries", [])
            except Exception:
                return []

    # ------------------------------------------------------------
    # CHANNEL INDEX (ALL VIDEOS)
    # ------------------------------------------------------------
    def fetch_channel_index(self, channel_url: str) -> List[Dict[str, Any]]:
        info = self._extract(channel_url)
        channel_id = info.get("channel_id")
        channel_name = info.get("channel")
        base = info.get("channel_url") or channel_url.rstrip("/")

        video_entries = self._fetch_tab(base, "videos")
        stream_entries = self._fetch_tab(base, "streams")
        live_entries = self._fetch_tab(base, "live")

        all_entries = video_entries + stream_entries + live_entries

        records = []
        seen = set()

        for e in all_entries:
            video_id = e.get("id")
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)

            ts = e.get("timestamp")
            publish_date = None
            if ts:
                publish_date = datetime.utcfromtimestamp(ts).date().isoformat()

            url = e.get("url") or f"https://www.youtube.com/watch?v={video_id}"

            records.append({
                "video_id": video_id,
                "title": e.get("title", ""),
                "publish_date": publish_date,
                "duration": e.get("duration"),
                "original_url": url,
                "channel_id": channel_id,
                "channel_name": channel_name,
            })

        return records

    # ------------------------------------------------------------
    # DOWNLOAD AUDIO (delegates to your unified downloader)
    # ------------------------------------------------------------
    def download_audio(self, url: str, out_dir: Path, title: str) -> Path:
        """
        Wrapper around your unified download_mp3() function.
        """
        return download_mp3(url, str(out_dir), title)

    # ------------------------------------------------------------
    # NORMALIZE (FINAL INGESTION SCHEMA)
    # ------------------------------------------------------------
    def normalize(self, video_record: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
        """
        Works for both parse_video() and channel_index records.
        """
        return {
            "video_id": video_record["video_id"],
            "title": video_record["title"],
            "channel_id": video_record["channel_id"],
            "channel_name": video_record["channel_name"],
            "channel_url": video_record.get("channel_url"),
            "publish_date": video_record.get("publish_date"),
            "description": video_record.get("description"),
            "duration": video_record.get("duration"),
            "thumbnail": video_record.get("thumbnail"),
            "audio_path": audio_path,
            "original_url": video_record["original_url"],
        }



class YouTubeAudioDataSource(DataSourceBase):
    """
    Minimal ingestion datasource for YouTube audio.
    """

    def __init__(
        self,
        name: str,
        adapter: Any,
        log_path: str,
        index_record_path: str,
        base_dir: str,
        config: Dict[str, Any] = None,
    ):
        super().__init__(name=name, log_path=log_path, config=config)
        self.adapter = adapter
        self.index_record_path = index_record_path
        self.base_dir = Path(base_dir)

        # --- Ensure log file exists and contains valid JSON list ---
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not log_file.exists():
            fs.write_json(log_file, [])   # MUST be a list

        # --- Ensure index file exists and contains valid JSON dict ---
        index_file = Path(index_record_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        if not index_file.exists():
            fs.write_json(index_file, {})   # MUST be a dict

        # --- Ensure base_dir exists ---
        self.base_dir.mkdir(parents=True, exist_ok=True)


    def ingest_log(self):
        """
        Ensure every log entry has a deterministic, URL-based 'id'.
        """
        if not self._logs:
            return

        normalized = []
        for entry in self._logs:
            if "id" not in entry:
                # Extract the URL from the log entry
                url = None
                try:
                    url = entry["payload"]["metadata"]["original_url"]
                except Exception:
                    # fallback: hash the whole entry
                    raw = json.dumps(entry, sort_keys=True)
                    entry_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                    entry = {**entry, "id": entry_id}
                    normalized.append(entry)
                    continue

                # Deterministic hash of the URL
                entry_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
                entry = {**entry, "id": entry_id}

            normalized.append(entry)

        fs.upsert_json_records(
            self.log_path,
            normalized,
            key="id",
        )

    # ------------------------------------------------------------
    # FETCH (PURE TASK SELECTION)
    # ------------------------------------------------------------
    def fetch(self, mode: str, **kwargs) -> List[Dict[str, Any]]:
        if mode == "single_video":
            url = kwargs.get("url")
            if not url:
                raise ValueError("single_video mode requires url")
            return [self._make_task_from_url(url)]

        if mode == "channel_index":
            url = kwargs.get("url")
            if not url:
                raise ValueError("channel_index mode requires url")
            return self._refresh_channel_index(url)

        if mode == "video_from_index":
            video_id = kwargs.get("video_id")
            if not video_id:
                raise ValueError("video_from_index mode requires video_id")
            return [self._make_task_from_index(video_id)]

        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------
    # TASK BUILDERS
    # ------------------------------------------------------------
    def _make_task_from_url(self, url: str) -> Dict[str, Any]:
        info = self.adapter.parse_video(url)
        video_id = info["video_id"]

        event_dir = str(self.base_dir / video_id)

        return {
            "status": "pending",
            "metadata": {
                "video_id": video_id,
                "original_url": url,
                "channel_id": info["channel_id"],
                "title": info["title"],
            },
            "destination_info": {
                "event_dir": event_dir,
                "index_store_path": self.index_record_path,
            },
        }

    def _make_task_from_index(self, video_id: str) -> Dict[str, Any]:
        index_file = Path(self.index_record_path)
        index_records = fs.read_json(index_file) or {}

        record = index_records.get(video_id)
        if not record:
            raise ValueError(f"Video {video_id} not found in index")

        event_dir = str(self.base_dir / video_id)

        return {
            "status": "pending",
            "metadata": {
                "video_id": video_id,
                "original_url": record["original_url"],
                "channel_id": record["channel_id"],
                "title": record["title"],
            },
            "destination_info": {
                "event_dir": event_dir,
                "index_store_path": self.index_record_path,
            },
        }

    # ------------------------------------------------------------
    # CHANNEL INDEX (DISCOVERY ONLY)
    # ------------------------------------------------------------
    def _refresh_channel_index(self, url: str) -> List[Dict[str, Any]]:
        records = self.adapter.fetch_channel_index(url)

        index_file = Path(self.index_record_path)
        index_records = fs.read_json(index_file) or {}

        for r in records:
            index_records[r["video_id"]] = r

        fs.write_json(index_file, jsonable(index_records))

        return []  # discovery only

    # ------------------------------------------------------------
    # APPLY (REAL INGESTION)
    # ------------------------------------------------------------
    def apply_update(self):
        metadata = self.context.metadata
        dest = self.context.destination_info

        video_id = metadata["video_id"]
        url = metadata["original_url"]
        title = metadata["title"]

        event_dir = Path(dest["event_dir"])
        event_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download audio
        audio_path = self.adapter.download_audio(url, event_dir, title)

        # 2. Parse full metadata
        video_info = self.adapter.parse_video(url)

        # 3. Normalize
        normalized = self.adapter.normalize(video_info, str(audio_path))

        # 4. Save normalized only
        fs.write_json(event_dir / "normalized.json", jsonable(normalized))

        # ------------------------------------------------------------
        # NEW: Log destination_info explicitly
        # ------------------------------------------------------------
        self.add_log(
            make_log_entry(
                self.name,
                "applied",
                {
                    "metadata": metadata,
                    "destination_info": dest,
                }
            )
        )

        return {
            "id": video_id,
            "video_id": video_id,
            "event_dir": str(event_dir),
            "original_url": url,
            "channel_id": video_info["channel_id"],
        }



"""
consider using the api

from yt_dlp import YoutubeDL

def download_mp3(url: str, out_dir: str, title: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(title)
    mp3_path = out_dir / f"{safe_title}.mp3"

    if mp3_path.exists():
        return mp3_path

    ydl_opts = {
        "outtmpl": str(out_dir / f"{safe_title}.%(ext)s"),
        "format": "bestaudio/best",
        "cookiesfrombrowser": "chrome",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "forceipv4": True,
        "postprocessors": [],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded = next(out_dir.glob(f"{safe_title}.*"))

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(downloaded),
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "48k",
        "-ar", "22050",
        "-ac", "1",
        str(mp3_path)
    ], check=True)

    downloaded.unlink()
    return mp3_path

"""