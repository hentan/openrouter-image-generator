"""Application-level operations shared by the desktop UI and its tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import requests

from openrouter_client import ImageAsset, ImageModel, OpenRouterClient, OpenRouterError


class ImageGeneratorService:
    def __init__(
        self,
        client_factory: Callable[..., OpenRouterClient] = OpenRouterClient,
        download: Callable[..., requests.Response] = requests.get,
    ) -> None:
        self._client_factory = client_factory
        self._download = download

    def list_models(self, api_key: str) -> list[ImageModel]:
        return self._client_factory(api_key).list_image_models()

    def enhance_prompt(self, api_key: str, model: str, description: str) -> str:
        return self._client_factory(api_key).enhance_prompt(description, model)

    def generate_image(self, api_key: str, model: str, prompt: str) -> ImageAsset:
        image = self._client_factory(api_key).generate_image(prompt, model)
        if image.data is not None or not image.url:
            return image

        try:
            response = self._download(image.url, timeout=45)
        except requests.RequestException as error:
            raise OpenRouterError(f"Не удалось загрузить изображение: {error}") from error
        if response.status_code >= 400:
            raise OpenRouterError(
                f"Не удалось загрузить изображение (HTTP {response.status_code})."
            )
        mime_type = response.headers.get("Content-Type", image.mime_type).split(";", 1)[0]
        return ImageAsset(data=response.content, mime_type=mime_type)

    @staticmethod
    def save_image(image: ImageAsset, destination: str | Path) -> Path:
        if image.data is None:
            raise OpenRouterError("Нет данных изображения для сохранения.")
        path = Path(destination)
        try:
            path.write_bytes(image.data)
        except OSError as error:
            raise OpenRouterError(f"Не удалось сохранить изображение: {error}") from error
        return path
