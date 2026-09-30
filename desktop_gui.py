"""Tkinter desktop interface for OpenRouter image generation."""

from __future__ import annotations

import io
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Callable

from PIL import Image, ImageTk, UnidentifiedImageError

from image_service import ImageGeneratorService
from openrouter_client import ImageAsset, ImageModel


class ImageGeneratorWindow:
    def __init__(self, root: tk.Tk, service: ImageGeneratorService | None = None) -> None:
        self.root = root
        self.service = service or ImageGeneratorService()
        self.root.title("OpenRouter — генерация изображений")
        self.root.geometry("850x760")
        self.root.minsize(680, 620)

        self.models: dict[str, str] = {}
        self.completed_tasks: queue.Queue[tuple[ttk.Button, Callable[..., Any], Any, Exception | None]] = queue.Queue()
        self.preview_image: ImageTk.PhotoImage | None = None
        self.current_image: ImageAsset | None = None

        self._build_ui()
        self.root.after(100, self._process_completed_tasks)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=18)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Генератор изображений", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            main,
            text="Выберите модель OpenRouter, опишите изображение и при желании улучшите промпт.",
        ).pack(anchor="w", pady=(2, 14))

        settings = ttk.LabelFrame(main, text="Настройки OpenRouter", padding=12)
        settings.pack(fill="x", pady=(0, 12))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="API-ключ:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.api_key = tk.StringVar(value=os.getenv("OPENROUTER_API_KEY", ""))
        ttk.Entry(settings, textvariable=self.api_key, show="*").grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=5
        )

        ttk.Label(settings, text="Модель для улучшения промпта:").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=5
        )
        self.prompt_model = tk.StringVar(
            value=os.getenv("OPENROUTER_PROMPT_MODEL", "openai/gpt-4o-mini")
        )
        ttk.Entry(settings, textvariable=self.prompt_model).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=5
        )

        ttk.Label(settings, text="Модель изображения:").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=5
        )
        self.model_choice = tk.StringVar()
        self.model_combo = ttk.Combobox(
            settings, textvariable=self.model_choice, state="readonly"
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", pady=5)
        self.load_models_button = ttk.Button(
            settings, text="Загрузить модели", command=self.load_models
        )
        self.load_models_button.grid(row=2, column=2, sticky="e", padx=(8, 0), pady=5)

        prompt_frame = ttk.LabelFrame(main, text="Описание изображения", padding=10)
        prompt_frame.pack(fill="x", pady=(0, 10))
        self.prompt_text = tk.Text(
            prompt_frame,
            height=5,
            wrap="word",
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=8,
        )
        self.prompt_text.pack(fill="x")
        self.prompt_text.insert("1.0", "")
        self.prompt_text.configure(
            insertbackground="#20242b",
            background="#ffffff",
            foreground="#20242b",
        )

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(0, 8))
        self.enhance_button = ttk.Button(
            actions, text="Улучшить промпт", command=self.enhance_prompt
        )
        self.enhance_button.pack(side="left")
        self.generate_button = ttk.Button(
            actions, text="Сгенерировать изображение", command=self.generate_image
        )
        self.generate_button.pack(side="left", padx=(8, 0))
        self.save_button = ttk.Button(
            actions,
            text="Сохранить изображение",
            command=self.save_image,
            state="disabled",
        )
        self.save_button.pack(side="left", padx=(8, 0))

        self.status = tk.StringVar(value="Введите API-ключ и загрузите список моделей.")
        self.status_label = ttk.Label(main, textvariable=self.status, wraplength=800)
        self.status_label.pack(anchor="w", pady=(0, 8))

        preview_frame = ttk.LabelFrame(main, text="Предпросмотр", padding=10)
        preview_frame.pack(fill="both", expand=True)
        self.preview_label = ttk.Label(
            preview_frame,
            text="Здесь появится сгенерированное изображение",
            anchor="center",
        )
        self.preview_label.pack(fill="both", expand=True)

    def _get_api_key(self) -> str:
        key = self.api_key.get().strip()
        if not key:
            self._set_status("Введите API-ключ OpenRouter.", error=True)
        return key

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status.set(message)
        self.status_label.configure(foreground="#b42318" if error else "#344054")

    def _run_async(
        self,
        button: ttk.Button,
        progress_message: str,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        button.configure(state="disabled")
        self._set_status(progress_message)

        def worker() -> None:
            try:
                result = task()
                error = None
            except Exception as caught_error:  # Display unexpected API and decoding errors in the UI.
                result = None
                error = caught_error
            self.completed_tasks.put((button, on_success, result, error))

        threading.Thread(target=worker, daemon=True).start()

    def _process_completed_tasks(self) -> None:
        while True:
            try:
                button, on_success, result, error = self.completed_tasks.get_nowait()
            except queue.Empty:
                break
            button.configure(state="normal")
            if error is not None:
                self._set_status(str(error), error=True)
            else:
                on_success(result)
        self.root.after(100, self._process_completed_tasks)

    def load_models(self) -> None:
        key = self._get_api_key()
        if not key:
            return

        def loaded(models: list[ImageModel]) -> None:
            self.models = {f"{model.name} ({model.id})": model.id for model in models}
            self.model_combo["values"] = list(self.models)
            if models:
                self.model_choice.set(next(iter(self.models)))
                self._set_status(f"Загружено моделей: {len(models)}.")
            else:
                self.model_choice.set("")
                self._set_status("OpenRouter не нашёл модели с генерацией изображений.", error=True)

        self._run_async(
            self.load_models_button,
            "Загружаю список моделей...",
            lambda: self.service.list_models(key),
            loaded,
        )

    def enhance_prompt(self) -> None:
        key = self._get_api_key()
        description = self.prompt_text.get("1.0", "end").strip()
        prompt_model = self.prompt_model.get().strip()
        if not key:
            return
        if not description:
            self._set_status("Сначала введите описание изображения.", error=True)
            return
        if not prompt_model:
            self._set_status("Укажите текстовую модель для улучшения промпта.", error=True)
            return

        def improved(prompt: str) -> None:
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", prompt)
            self._set_status("Промпт улучшен — при необходимости отредактируйте его.")

        self._run_async(
            self.enhance_button,
            "Улучшаю описание...",
            lambda: self.service.enhance_prompt(key, prompt_model, description),
            improved,
        )

    def generate_image(self) -> None:
        key = self._get_api_key()
        prompt = self.prompt_text.get("1.0", "end").strip()
        model_id = self.models.get(self.model_choice.get())
        if not key:
            return
        if not model_id:
            self._set_status("Сначала загрузите и выберите модель изображения.", error=True)
            return
        if not prompt:
            self._set_status("Сначала введите описание изображения.", error=True)
            return

        self._run_async(
            self.generate_button,
            "Создаю изображение...",
            lambda: self.service.generate_image(key, model_id, prompt),
            self._show_image,
        )

    def _show_image(self, image: ImageAsset) -> None:
        if image.data is None:
            self._set_status("В ответе не удалось получить данные изображения.", error=True)
            return
        try:
            picture = Image.open(io.BytesIO(image.data))
            picture.thumbnail((760, 440), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(picture)
        except (UnidentifiedImageError, OSError) as error:
            self._set_status(f"Не удалось открыть изображение: {error}", error=True)
            return
        self.preview_label.configure(image=self.preview_image, text="")
        self.current_image = image
        self.save_button.configure(state="normal")
        self._set_status("Изображение готово.")

    def save_image(self) -> None:
        if self.current_image is None or self.current_image.data is None:
            self._set_status("Сначала сгенерируйте изображение.", error=True)
            return

        extensions = {
            "image/jpeg": (".jpg", "JPEG"),
            "image/png": (".png", "PNG"),
            "image/webp": (".webp", "WebP"),
            "image/gif": (".gif", "GIF"),
            "image/bmp": (".bmp", "BMP"),
        }
        extension, format_name = extensions.get(self.current_image.mime_type, (".png", "PNG"))
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить изображение",
            initialfile=f"generated-image{extension}",
            defaultextension=extension,
            filetypes=[(f"{format_name} image", f"*{extension}")],
        )
        if not destination:
            return

        try:
            saved_path = self.service.save_image(self.current_image, destination)
        except Exception as error:
            self._set_status(str(error), error=True)
            return
        self._set_status(f"Изображение сохранено: {saved_path}")


def main() -> None:
    root = tk.Tk()
    ImageGeneratorWindow(root)
    root.mainloop()
