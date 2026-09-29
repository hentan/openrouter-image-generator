import base64
from unittest.mock import Mock

import pytest
import requests

from openrouter_client import ImageAsset, OpenRouterClient, OpenRouterError


def mock_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = ""
    return response


def test_list_image_models_filters_models_and_sorts_by_name():
    session = Mock()
    session.request.return_value = mock_response(
        {
            "data": [
                {"id": "vendor/z-image", "name": "Z Image", "architecture": {"output_modalities": ["text", "image"]}},
                {"id": "vendor/text", "name": "Text only", "architecture": {"output_modalities": ["text"]}},
                {"id": "vendor/a-image", "name": "A Image", "architecture": {"modality": "text->image"}},
            ]
        }
    )
    client = OpenRouterClient("test-key", session=session)

    models = client.list_image_models()

    assert [model.id for model in models] == ["vendor/a-image", "vendor/z-image"]
    request = session.request.call_args
    assert request.args[:2] == ("GET", "https://openrouter.ai/api/v1/models")
    assert request.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_generate_image_decodes_data_uri_from_message_images():
    raw_image = b"fake-png-data"
    encoded_image = base64.b64encode(raw_image).decode("ascii")
    session = Mock()
    session.request.return_value = mock_response(
        {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                        ]
                    }
                }
            ]
        }
    )
    client = OpenRouterClient("test-key", session=session)

    image = client.generate_image("a cat", "vendor/image-model")

    assert image == ImageAsset(data=raw_image, mime_type="image/png")
    request_payload = session.request.call_args.kwargs["json"]
    assert request_payload["modalities"] == ["image", "text"]
    assert request_payload["model"] == "vendor/image-model"


def test_generate_image_accepts_remote_image_url_from_content_parts():
    session = Mock()
    session.request.return_value = mock_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Done"},
                            {"type": "image_url", "image_url": {"url": "https://images.example/result.png"}},
                        ]
                    }
                }
            ]
        }
    )

    image = OpenRouterClient("test-key", session=session).generate_image("a cat", "vendor/model")

    assert image == ImageAsset(url="https://images.example/result.png")


def test_enhance_prompt_sends_original_description_and_returns_text():
    session = Mock()
    session.request.return_value = mock_response(
        {"choices": [{"message": {"content": "Подробное описание кота при мягком свете"}}]}
    )
    client = OpenRouterClient("test-key", session=session)

    result = client.enhance_prompt("кот", "openai/gpt-4o-mini")

    assert result == "Подробное описание кота при мягком свете"
    payload = session.request.call_args.kwargs["json"]
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["messages"][1]["content"] == "кот"


def test_api_error_includes_openrouter_message():
    session = Mock()
    session.request.return_value = mock_response(
        {"error": {"message": "Invalid API key"}}, status_code=401
    )
    client = OpenRouterClient("bad-key", session=session)

    with pytest.raises(OpenRouterError, match="Invalid API key"):
        client.list_image_models()


def test_request_connection_error_is_reported_as_openrouter_error():
    session = Mock()
    session.request.side_effect = requests.ConnectionError("offline")
    client = OpenRouterClient("test-key", session=session)

    with pytest.raises(OpenRouterError, match="Не удалось подключиться"):
        client.list_image_models()


def test_generate_image_rejects_text_only_response():
    session = Mock()
    session.request.return_value = mock_response(
        {"choices": [{"message": {"content": "Here is your image."}}]}
    )

    with pytest.raises(OpenRouterError, match="В ответе нет изображения"):
        OpenRouterClient("test-key", session=session).generate_image("a cat", "vendor/model")
