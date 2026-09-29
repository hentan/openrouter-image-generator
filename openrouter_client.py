"""Small OpenRouter API client used by the desktop interface."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_to_bytes

import requests


class OpenRouterError(Exception):
    """An API or response error that can be shown to the user."""


@dataclass(frozen=True)
class ImageModel:
    id: str
    name: str


@dataclass(frozen=True)
class ImageAsset:
    """An image returned by OpenRouter, either as bytes or as a remote URL."""

    data: bytes | None = None
    url: str | None = None
    mime_type: str = "image/png"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 90,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def list_image_models(self) -> list[ImageModel]:
        response = self._request("GET", f"{self.base_url}/models")
        payload = self._json(response)
        records = payload.get("data")
        if not isinstance(records, list):
            raise OpenRouterError("OpenRouter вернул некорректный список моделей.")

        models: list[ImageModel] = []
        for record in records:
            if not isinstance(record, dict) or not self._supports_image_output(record):
                continue
            model_id = record.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            name = record.get("name")
            models.append(ImageModel(model_id, name if isinstance(name, str) else model_id))
        return sorted(models, key=lambda model: model.name.casefold())

    def enhance_prompt(self, description: str, model: str) -> str:
        if not description.strip():
            raise OpenRouterError("Сначала введите описание изображения.")

        response = self._request(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's short image description as a vivid, detailed prompt "
                            "for an image generation model. Preserve the original subject and intent; "
                            "add useful visual details such as composition, lighting, colors, style, "
                            "and environment without inventing unrelated elements. Use the same "
                            "language as the user. Return only the finished prompt, with no preamble."
                        ),
                    },
                    {"role": "user", "content": description.strip()},
                ],
            },
        )
        payload = self._json(response)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenRouterError("Модель не вернула улучшенный промпт.") from error

        prompt = self._text_content(content).strip()
        if not prompt:
            raise OpenRouterError("Модель вернула пустой промпт.")
        return prompt

    def generate_image(self, prompt: str, model: str) -> ImageAsset:
        if not prompt.strip():
            raise OpenRouterError("Сначала введите описание изображения.")

        response = self._request(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt.strip()}],
                "modalities": ["image", "text"],
            },
        )
        payload = self._json(response)
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenRouterError("OpenRouter не вернул результат генерации.") from error

        image = self._find_image(message)
        if image is None:
            raise OpenRouterError(
                "В ответе нет изображения. Проверьте, что выбранная модель поддерживает "
                "генерацию изображений."
            )
        return image

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}))
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.setdefault("Content-Type", "application/json")
        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as error:
            raise OpenRouterError(f"Не удалось подключиться к OpenRouter: {error}") from error

        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                error_data = body.get("error", body) if isinstance(body, dict) else body
                if isinstance(error_data, dict):
                    detail = str(error_data.get("message") or error_data.get("detail") or "")
                else:
                    detail = str(error_data)
            except (ValueError, TypeError):
                detail = response.text[:500]
            message = f"OpenRouter вернул ошибку {response.status_code}"
            if detail:
                message += f": {detail}"
            raise OpenRouterError(message)
        return response

    @staticmethod
    def _json(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except (ValueError, TypeError) as error:
            raise OpenRouterError("OpenRouter вернул ответ в неизвестном формате.") from error
        if not isinstance(payload, dict):
            raise OpenRouterError("OpenRouter вернул ответ в неизвестном формате.")
        return payload

    @staticmethod
    def _supports_image_output(record: dict[str, Any]) -> bool:
        architecture = record.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}
        output_modalities = record.get("output_modalities") or architecture.get(
            "output_modalities"
        )
        if isinstance(output_modalities, list) and any(
            isinstance(item, str) and item.lower() == "image" for item in output_modalities
        ):
            return True
        modality = architecture.get("modality") or record.get("modality")
        return isinstance(modality, str) and bool(
            re.search(r"(?:^|[+>,])\s*image\s*$|->\s*image\s*$", modality, re.IGNORECASE)
        )

    @classmethod
    def _find_image(cls, message: dict[str, Any]) -> ImageAsset | None:
        images = message.get("images")
        if isinstance(images, list):
            for image in images:
                asset = cls._image_from_part(image)
                if asset:
                    return asset

        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                asset = cls._image_from_part(part)
                if asset:
                    return asset
        elif isinstance(content, str):
            data_uri_match = re.search(r"data:image/[^\s)]+", content)
            if data_uri_match:
                asset = cls._asset_from_url(data_uri_match.group(0).rstrip("`\"'"))
                if asset:
                    return asset
            markdown_match = re.search(r"!\[[^]]*\]\((https?://[^)]+)\)", content)
            if markdown_match:
                return ImageAsset(url=markdown_match.group(1))
        return None

    @classmethod
    def _image_from_part(cls, part: Any) -> ImageAsset | None:
        if not isinstance(part, dict):
            return None
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str):
            asset = cls._asset_from_url(image_url)
            if asset:
                return asset

        image = part.get("image")
        if isinstance(image, str):
            asset = cls._asset_from_url(image)
            if asset:
                return asset
        elif isinstance(image, dict):
            url = image.get("url") or image.get("data")
            if isinstance(url, str):
                asset = cls._asset_from_url(url)
                if asset:
                    return asset
        return None

    @staticmethod
    def _asset_from_url(value: str) -> ImageAsset | None:
        if value.startswith("data:image/"):
            header, separator, encoded = value.partition(",")
            if not separator:
                return None
            mime_type = header[5:].split(";", 1)[0] or "image/png"
            try:
                data = (
                    base64.b64decode(encoded, validate=True)
                    if ";base64" in header
                    else unquote_to_bytes(encoded)
                )
            except (binascii.Error, ValueError):
                return None
            return ImageAsset(data=data, mime_type=mime_type)
        if value.startswith(("https://", "http://")):
            return ImageAsset(url=value)
        return None

    @staticmethod
    def _text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""
