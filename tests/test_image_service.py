from unittest.mock import Mock

import requests
import pytest

from image_service import ImageGeneratorService
from openrouter_client import ImageAsset, ImageModel, OpenRouterError


class FakeClient:
    def __init__(self, api_key, image=ImageAsset(data=b"image-bytes")):
        self.api_key = api_key
        self.image = image

    def list_image_models(self):
        return [ImageModel("test/image", "Test Image")]

    def enhance_prompt(self, description, model):
        return f"{description} — detailed"

    def generate_image(self, prompt, model):
        return self.image


def test_service_uses_openrouter_client_for_models_and_prompt_enhancement():
    service = ImageGeneratorService(client_factory=FakeClient)

    models = service.list_models("secret")
    prompt = service.enhance_prompt("secret", "text/model", "a fox")

    assert models == [ImageModel("test/image", "Test Image")]
    assert prompt == "a fox — detailed"


def test_service_returns_image_bytes_without_extra_download():
    download = Mock()
    service = ImageGeneratorService(
        client_factory=lambda api_key: FakeClient(api_key, ImageAsset(data=b"png")),
        download=download,
    )

    image = service.generate_image("secret", "image/model", "a fox")

    assert image.data == b"png"
    download.assert_not_called()


def test_service_downloads_remote_image_for_local_preview():
    response = Mock(status_code=200, content=b"downloaded-image")
    response.headers = {"Content-Type": "image/jpeg; charset=binary"}
    download = Mock(return_value=response)
    service = ImageGeneratorService(
        client_factory=lambda api_key: FakeClient(
            api_key, ImageAsset(url="https://images.example/generated.jpg")
        ),
        download=download,
    )

    image = service.generate_image("secret", "image/model", "a fox")

    assert image == ImageAsset(data=b"downloaded-image", mime_type="image/jpeg")
    download.assert_called_once_with("https://images.example/generated.jpg", timeout=45)


def test_service_reports_remote_image_download_errors():
    download = Mock(side_effect=requests.ConnectionError("offline"))
    service = ImageGeneratorService(
        client_factory=lambda api_key: FakeClient(
            api_key, ImageAsset(url="https://images.example/generated.jpg")
        ),
        download=download,
    )

    with pytest.raises(OpenRouterError, match="Не удалось загрузить изображение"):
        service.generate_image("secret", "image/model", "a fox")


def test_service_saves_image_bytes_to_selected_path(tmp_path):
    destination = tmp_path / "generated.png"
    image = ImageAsset(data=b"png-content", mime_type="image/png")

    saved_path = ImageGeneratorService.save_image(image, destination)

    assert saved_path == destination
    assert destination.read_bytes() == b"png-content"


def test_service_refuses_to_save_image_without_data(tmp_path):
    with pytest.raises(OpenRouterError, match="Нет данных изображения"):
        ImageGeneratorService.save_image(ImageAsset(url="https://example.com/image.png"), tmp_path / "image.png")
