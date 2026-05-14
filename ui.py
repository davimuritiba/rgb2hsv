from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from processing import carregar_rgb_do_disco, hsv_para_preview_matiz, rgb_para_hsv

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Área máxima por painel (largura, altura) — duas colunas lado a lado
PAINEL_MAX = (520, 420)


def redimensionar_para_caber(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    w, h = img.size
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Conversor RGB → HSV")
        self.minsize(960, 560)
        self.geometry("1100x640")

        self._rgb: np.ndarray | None = None
        self._hsv: np.ndarray | None = None
        self._caminho: str | None = None
        self._img_rgb: ctk.CTkImage | None = None
        self._img_hsv: ctk.CTkImage | None = None

        self.configure(fg_color=("#0f1419", "#0f1419"))

        self._build_header()
        self._build_toolbar()
        self._build_preview_area()
        self._build_footer()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill=tk.X, padx=28, pady=(24, 8))

        ctk.CTkLabel(
            header,
            text="RGB → HSV",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=("#e8eef5", "#e8eef5"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Abra uma imagem do disco. À esquerda: RGB original. À direita: mapa de matiz (H) do HSV.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=("#8b9cb3", "#8b9cb3"),
        ).pack(anchor="w", pady=(4, 0))

    def _build_toolbar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=("#1a222d", "#1a222d"), corner_radius=12)
        bar.pack(fill=tk.X, padx=24, pady=(12, 8))

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill=tk.X, padx=14, pady=12)

        self.btn_abrir = ctk.CTkButton(
            inner,
            text="Abrir imagem…",
            width=150,
            height=38,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.abrir_imagem,
        )
        self.btn_abrir.pack(side=tk.LEFT)

        self.btn_salvar = ctk.CTkButton(
            inner,
            text="Salvar HSV…",
            width=150,
            height=38,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=self.salvar_hsv,
            state="disabled",
        )
        self.btn_salvar.pack(side=tk.LEFT, padx=(12, 0))

    def _build_preview_area(self) -> None:
        outer = ctk.CTkFrame(self, fg_color=("#151c26", "#151c26"), corner_radius=14)
        outer.pack(fill=tk.BOTH, expand=True, padx=24, pady=(4, 8))

        row = ctk.CTkFrame(outer, fg_color="transparent")
        row.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.col_rgb = ctk.CTkFrame(row, fg_color=("#1e2733", "#1e2733"), corner_radius=12)
        self.col_rgb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.col_hsv = ctk.CTkFrame(row, fg_color=("#1e2733", "#1e2733"), corner_radius=12)
        self.col_hsv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        ctk.CTkLabel(
            self.col_rgb,
            text="Entrada (RGB)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(pady=(12, 6))

        self.lbl_rgb = ctk.CTkLabel(self.col_rgb, text="", fg_color=("#0d1117", "#0d1117"), corner_radius=8)
        self.lbl_rgb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        ctk.CTkLabel(
            self.col_hsv,
            text="HSV — mapa de matiz (H)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#c4a8e0", "#c4a8e0"),
        ).pack(pady=(12, 6))

        self.lbl_hsv = ctk.CTkLabel(self.col_hsv, text="", fg_color=("#0d1117", "#0d1117"), corner_radius=8)
        self.lbl_hsv.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._set_placeholders()

    def _set_placeholders(self) -> None:
        for lbl in (self.lbl_rgb, self.lbl_hsv):
            lbl.configure(
                text="Nenhuma imagem",
                image=None,
                font=ctk.CTkFont(size=15),
                text_color=("#5c6b80", "#5c6b80"),
            )

    def _build_footer(self) -> None:
        self.lbl_info = ctk.CTkLabel(
            self,
            text="Carregue uma imagem para começar.",
            font=ctk.CTkFont(size=12),
            text_color=("#6b7c93", "#6b7c93"),
            anchor="w",
        )
        self.lbl_info.pack(fill=tk.X, padx=28, pady=(0, 18))

    def abrir_imagem(self) -> None:
        caminho = filedialog.askopenfilename(
            parent=self,
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not caminho:
            return

        try:
            rgb = carregar_rgb_do_disco(caminho)
            hsv = rgb_para_hsv(rgb)
            matiz_rgb = hsv_para_preview_matiz(hsv)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro ao abrir imagem", str(exc), parent=self)
            return

        self._caminho = caminho
        self._rgb = rgb
        self._hsv = hsv

        pil_rgb = Image.fromarray(rgb)
        pil_hsv = Image.fromarray(matiz_rgb)
        pil_rgb_r = redimensionar_para_caber(pil_rgb, PAINEL_MAX)
        pil_hsv_r = redimensionar_para_caber(pil_hsv, PAINEL_MAX)

        w1, h1 = pil_rgb_r.size
        w2, h2 = pil_hsv_r.size

        self._img_rgb = ctk.CTkImage(light_image=pil_rgb_r, dark_image=pil_rgb_r, size=(w1, h1))
        self._img_hsv = ctk.CTkImage(light_image=pil_hsv_r, dark_image=pil_hsv_r, size=(w2, h2))

        self.lbl_rgb.configure(image=self._img_rgb, text="")
        self.lbl_hsv.configure(image=self._img_hsv, text="")

        self.lbl_info.configure(
            text=f"Arquivo: {caminho}  ·  RGB {rgb.shape} → HSV {hsv.shape}",
            text_color=("#8b9cb3", "#8b9cb3"),
        )
        self.btn_salvar.configure(state="normal")

    def salvar_hsv(self) -> None:
        if self._hsv is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem antes.", parent=self)
            return

        caminho = filedialog.asksaveasfilename(
            parent=self,
            title="Salvar HSV",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not caminho:
            return

        ok = cv2.imwrite(caminho, self._hsv)
        if not ok:
            messagebox.showerror("Erro", "Não foi possível salvar o arquivo.", parent=self)
            return

        messagebox.showinfo("OK", f"HSV salvo em:\n{caminho}", parent=self)


def run() -> None:
    App().mainloop()
