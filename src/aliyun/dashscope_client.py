"""DashScope / Aliyun Model Studio API client utilities.

Provides a thin wrapper around DashScope async audio transcription and Qwen
chat completion, plus OSS upload for local files.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DASHSCOPE_API_V1 = "https://dashscope.aliyuncs.com/api/v1"
BASE_URL_V1 = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ASR_FILE_MODEL = "paraformer-v2"
DEFAULT_CHAT_MODEL = "qwen-plus"


def get_api_key() -> Optional[str]:
    return os.getenv("DASHSCOPE_API_KEY")


def upload_local_to_oss(api_key: str, file_path: Path, model: str = ASR_FILE_MODEL) -> str:
    """Upload a local file to DashScope temp OSS and return an oss:// URL."""
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    params = {"action": "getPolicy", "model": model}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    key = f"{data['upload_dir']}/{file_path.name}"

    with open(file_path, "rb") as f:
        files = {
            "OSSAccessKeyId": (None, data["oss_access_key_id"]),
            "Signature": (None, data["signature"]),
            "policy": (None, data["policy"]),
            "x-oss-object-acl": (None, data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_path.name, f),
        }
        r2 = requests.post(data["upload_host"], files=files, timeout=120)
    r2.raise_for_status()
    return f"oss://{key}"


def create_filetrans_task(
    api_key: str,
    file_url: str,
    enable_itn: bool = False,
    enable_words: bool = True,
    diarization_enabled: bool = True,
    speaker_count: Optional[int] = None,
) -> str:
    """Create an async file transcription task and return task_id."""
    url = f"{DASHSCOPE_API_V1}/services/audio/asr/transcription"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    if file_url.startswith("oss://"):
        headers["X-DashScope-OssResourceResolve"] = "enable"

    parameters: Dict[str, Any] = {
        "channel_id": [0],
        "enable_itn": enable_itn,
        "enable_words": enable_words,
    }
    if diarization_enabled:
        parameters["diarization_enabled"] = True
    if speaker_count is not None:
        parameters["speaker_count"] = speaker_count

    payload: Dict[str, Any] = {
        "model": ASR_FILE_MODEL,
        "parameters": parameters,
    }
    # paraformer-v2 and fun-asr models require file_urls array.
    # qwen3-asr-flash-filetrans uses file_url string.
    if ASR_FILE_MODEL in ("paraformer-v2", "fun-asr"):
        payload["input"] = {"file_urls": [file_url]}
    else:
        payload["input"] = {"file_url": file_url}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id in response: {data}")
    return task_id


def wait_filetrans_task(api_key: str, task_id: str, poll_interval: float = 5.0, timeout: float = 3600.0) -> Dict[str, Any]:
    """Poll a transcription task until completion or timeout."""
    url = f"{DASHSCOPE_API_V1}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        output = data.get("output", {})
        status = output.get("task_status")
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            return data
        time.sleep(poll_interval)
    raise TimeoutError(f"Transcription task {task_id} did not complete within {timeout}s")


def fetch_transcription_result(transcription_url: str) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch and parse the transcription result JSON."""
    r = requests.get(transcription_url, timeout=60)
    r.raise_for_status()
    data = r.json()
    sentences: List[Dict[str, Any]] = []
    transcripts = data.get("transcripts") or []
    for t in transcripts:
        for s in (t.get("sentences") or []):
            sentences.append({
                "text": s.get("text", ""),
                "begin_time": s.get("begin_time", 0),
                "end_time": s.get("end_time", 0),
                "speaker_id": s.get("speaker_id", "unknown"),
            })
    plain_text = " ".join(s.get("text", "") for s in sentences).strip()
    return sentences, plain_text


def run_filetrans(
    file_path: str | Path,
    enable_itn: bool = False,
    enable_words: bool = True,
    diarization_enabled: bool = True,
    speaker_count: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end: upload local file, transcribe with speaker diarization, return parsed result."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_url = upload_local_to_oss(api_key, file_path)
    task_id = create_filetrans_task(
        api_key,
        file_url,
        enable_itn=enable_itn,
        enable_words=enable_words,
        diarization_enabled=diarization_enabled,
        speaker_count=speaker_count,
    )
    data = wait_filetrans_task(api_key, task_id)
    output = data.get("output", {})
    status = output.get("task_status")
    if status != "SUCCEEDED":
        raise RuntimeError(f"Transcription failed: {status} - {output}")

    transcription_url = output.get("transcription_url") or (
        (output.get("result") or {}).get("transcription_url")
    )
    # paraformer-v2 returns results array with transcription_url inside.
    if not transcription_url:
        results = output.get("results", [])
        if results:
            transcription_url = results[0].get("transcription_url") or (
                (results[0].get("output") or {}).get("transcription_url")
            )
    if transcription_url:
        sentences, plain_text = fetch_transcription_result(transcription_url)
    else:
        # Fallback: try to parse inline sentences if available.
        sentences = []
        for t in output.get("result", {}).get("transcripts", []):
            for s in t.get("sentences", []):
                sentences.append({
                    "text": s.get("text", ""),
                    "begin_time": s.get("begin_time", 0),
                    "end_time": s.get("end_time", 0),
                    "speaker_id": s.get("speaker_id", "unknown"),
                })
        plain_text = " ".join(s["text"] for s in sentences).strip()

    return {
        "task_id": task_id,
        "sentences": sentences,
        "plain_text": plain_text,
    }


def qwen_chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    """Call Qwen chat completion and return assistant content."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=BASE_URL_V1)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


__all__ = [
    "get_api_key",
    "upload_local_to_oss",
    "create_filetrans_task",
    "wait_filetrans_task",
    "fetch_transcription_result",
    "run_filetrans",
    "qwen_chat",
]
