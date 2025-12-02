"""Интеграция с Adobe PDF Services API для конвертации PDF в Excel.

Версия без явного шага OCR: используем REST Export PDF (PDF -> XLSX).
Adobe сам решает, нужен ли OCR и какие языки использовать.
"""

from __future__ import annotations

import io
import os
import re
import time
from typing import Optional, Dict, Any, List

import requests
import pandas as pd


class AdobePDFService:
    """Минимальный клиент Adobe PDF Services (REST Export PDF)."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self._client_id = client_id or os.getenv("ADOBE_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("ADOBE_CLIENT_SECRET")
        self._region = region or os.getenv("ADOBE_REGION", "US")

        if not self._client_id or not self._client_secret:
            raise ValueError(
                "Adobe API credentials обязательны. "
                "Установите ADOBE_CLIENT_ID и ADOBE_CLIENT_SECRET."
            )

    def _base_url(self) -> str:
        if self._region.upper() == "EU":
            return "https://pdf-services-eu.adobe.io"
        return "https://pdf-services.adobe.io"

    def _get_access_token(self) -> str:
        token_url = "https://pdf-services.adobe.io/token"
        resp = requests.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"client_id": self._client_id, "client_secret": self._client_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"]

    def _upload_asset(self, access_token: str, pdf_bytes: bytes) -> str:
        base_url = self._base_url()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-Key": self._client_id,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{base_url}/assets",
            headers=headers,
            json={"mediaType": "application/pdf"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        upload_uri = data.get("uploadUri")
        asset_id = data.get("assetID")
        if not upload_uri or not asset_id:
            raise RuntimeError(f"Не удалось получить uploadUri/assetID: {data}")

        put_resp = requests.put(
            upload_uri,
            headers={"Content-Type": "application/pdf"},
            data=pdf_bytes,
            timeout=120,
        )
        put_resp.raise_for_status()
        return asset_id

    def _start_export_job(self, access_token: str, asset_id: str) -> str:
        base_url = self._base_url()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-Key": self._client_id,
            "Content-Type": "application/json",
        }
        export_url = f"{base_url}/operation/exportpdf"
        payload = {"assetID": asset_id, "targetFormat": "xlsx"}
        resp = requests.post(export_url, headers=headers, json=payload, timeout=60)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Ошибка создания export job: {resp.status_code} {resp.text[:300]}"
            )

        location = resp.headers.get("Location", "")
        job_id = None

        if location:
            parts = location.strip("/").split("/")
            if "status" in parts:
                idx = parts.index("status")
                if idx > 0:
                    job_id = parts[idx - 1]
            else:
                job_id = parts[-1]

            if not job_id or job_id == "status":
                m = re.search(r"/exportpdf/([^/]+)/status", location)
                if m:
                    job_id = m.group(1)

        if not job_id:
            try:
                body = resp.json()
                job_id = body.get("jobId") or body.get("id") or body.get("job_id")
            except Exception:
                pass

        if not job_id:
            raise RuntimeError(f"Не удалось извлечь jobId из ответа: {resp.text[:400]}")

        return job_id

    def _wait_for_result(self, access_token: str, job_id: str) -> bytes:
        base_url = self._base_url()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-Key": self._client_id,
            "Content-Type": "application/json",
        }
        status_url = f"{base_url}/operation/exportpdf/{job_id}/status"
        max_wait = int(os.getenv("ADOBE_JOB_TIMEOUT", "600"))
        start = time.time()

        while time.time() - start < max_wait:
            resp = requests.get(status_url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "unknown")

            if status in ("done", "success"):
                download_uri = None
                for key in ("asset", "result"):
                    obj = data.get(key)
                    if isinstance(obj, dict):
                        for k in (
                            "downloadUri",
                            "download_uri",
                            "downloadURL",
                            "download_url",
                        ):
                            if obj.get(k):
                                download_uri = obj[k]
                                break
                    if download_uri:
                        break

                if not download_uri:
                    for k in (
                        "downloadUri",
                        "download_uri",
                        "downloadURL",
                        "download_url",
                    ):
                        if data.get(k):
                            download_uri = data[k]
                            break

                if not download_uri:
                    result_url = f"{base_url}/operation/exportpdf/{job_id}/result"
                    r2 = requests.get(result_url, headers=headers, timeout=30)
                    if r2.status_code in (301, 302, 303, 307, 308):
                        download_uri = r2.headers.get("Location")
                    elif r2.headers.get("Content-Type", "").startswith(
                        "application/vnd.openxmlformats"
                    ):
                        return r2.content

                if not download_uri:
                    raise RuntimeError(
                        f"downloadUri не найден в ответе статуса: {data}"
                    )

                file_resp = requests.get(download_uri, timeout=120)
                file_resp.raise_for_status()
                return file_resp.content

            if status in ("failed", "error"):
                err = data.get("error")
                raise RuntimeError(f"Adobe job failed: {err}")

            time.sleep(2)

        raise TimeoutError("Adobe API job timeout")

    def convert_pdf_to_excel_bytes(self, pdf_bytes: bytes) -> bytes:
        """Возвращает байты Excel (XLSX) для одного PDF."""
        access_token = self._get_access_token()
        asset_id = self._upload_asset(access_token, pdf_bytes)
        job_id = self._start_export_job(access_token, asset_id)
        return self._wait_for_result(access_token, job_id)

    def convert_pdf_to_sheets(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Конвертирует PDF в Excel и возвращает список листов:
        [{ "sheet_name": str, "columns": [...], "rows": [ {col: value, ...}, ...] }, ...]
        """
        excel_bytes = self.convert_pdf_to_excel_bytes(pdf_bytes)
        bio = io.BytesIO(excel_bytes)
        xls = pd.ExcelFile(bio, engine="openpyxl")

        sheets: List[Dict[str, Any]] = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
            if df.empty:
                continue
            df = df.astype(object).where(pd.notna(df), None)
            rows = df.to_dict(orient="records")
            sheets.append(
                {
                    "sheet_name": sheet_name,
                    "columns": list(df.columns),
                    "rows": rows,
                }
            )
        return sheets


