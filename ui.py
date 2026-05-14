from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from processing import (
    ajustar_hsv,
    aplicar_blur_gaussiano,
    aplicar_blur_mediano,
    carregar_rgb_do_disco,
    detectar_bordas_canny,
    detectar_bordas_sobel,
    gerar_histograma,
    hsv_para_canal_cinza,
    hsv_para_preview_matiz,
    hsv_para_preview_saturacao,
    hsv_para_preview_valor,
    rgb_para_hsv,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PAINEL_MAX = (340, 320)


def redimensionar_para_caber(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    w, h = img.size
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def array_para_ctk_image(arr: np.ndarray, max_size: tuple[int, int]) -> ctk.CTkImage:
    """Converte array numpy em CTkImage."""
    pil_img = Image.fromarray(arr)
    pil_img = redimensionar_para_caber(pil_img, max_size)
    w, h = pil_img.size
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Editor RGB → HSV Avançado")
        self.minsize(1400, 900)
        self.geometry("1500x950")

        self._rgb: np.ndarray | None = None
        self._hsv: np.ndarray | None = None
        self._hsv_editado: np.ndarray | None = None
        self._caminho: str | None = None

        # Estado dos sliders
        self._ajuste_h = 0
        self._ajuste_s = 1.0
        self._ajuste_v = 1.0
        self._atualizando_sliders = False

        self.configure(fg_color=("#0f1419", "#0f1419"))

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_header()
        self._build_main_content()
        self._build_footer()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill=tk.X, padx=20, pady=(16, 8))

        ctk.CTkLabel(
            header,
            text="Editor RGB → HSV Avançado",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=("#e8eef5", "#e8eef5"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Carregue uma imagem, ajuste canais HSV, visualize histogramas e aplique filtros.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=("#8b9cb3", "#8b9cb3"),
        ).pack(anchor="w", pady=(2, 0))

    def _build_main_content(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # Lado esquerdo: controles
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 12))

        self._build_controls(left)

        # Lado direito: visualizações
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_visualizations(right)

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        """Painel de controles à esquerda."""
        ctrl_frame = ctk.CTkFrame(parent, fg_color=("#151c26", "#151c26"), corner_radius=12)
        ctrl_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Seção: Arquivo
        ctk.CTkLabel(
            ctrl_frame,
            text="Arquivo",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(anchor="w", padx=14, pady=(14, 10))

        inner = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        inner.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.btn_abrir = ctk.CTkButton(
            inner,
            text="Abrir imagem…",
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.abrir_imagem,
        )
        self.btn_abrir.pack(fill=tk.X, padx=2, pady=(0, 6))

        self.btn_salvar = ctk.CTkButton(
            inner,
            text="Salvar imagem…",
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=self.salvar_hsv,
            state="disabled",
        )
        self.btn_salvar.pack(fill=tk.X, padx=2)

        # Separador
        ctk.CTkFrame(ctrl_frame, fg_color="#2a3540", height=1).pack(
            fill=tk.X, padx=12, pady=12
        )

        # Seção: Ajustes
        ctk.CTkLabel(
            ctrl_frame,
            text="Ajustes HSV",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(anchor="w", padx=14, pady=(0, 10))

        adjusts = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        adjusts.pack(fill=tk.X, padx=12, pady=(0, 12))

        # Slider H
        ctk.CTkLabel(
            adjusts,
            text="Matiz (H): 0°",
            font=ctk.CTkFont(size=11),
            text_color=("#c4d0e8", "#c4d0e8"),
        ).pack(anchor="w", pady=(0, 4))

        self.slider_h = ctk.CTkSlider(
            adjusts,
            from_=-180,
            to=180,
            number_of_steps=360,
            command=self._on_slider_changed,
            state="disabled",
        )
        self.slider_h.pack(fill=tk.X, pady=(0, 10))
        self.slider_h.set(0)

        # Slider S
        ctk.CTkLabel(
            adjusts,
            text="Saturação (S): 1.0x",
            font=ctk.CTkFont(size=11),
            text_color=("#c4d0e8", "#c4d0e8"),
        ).pack(anchor="w", pady=(0, 4))

        self.slider_s = ctk.CTkSlider(
            adjusts,
            from_=0.0,
            to=2.0,
            number_of_steps=200,
            command=self._on_slider_changed,
            state="disabled",
        )
        self.slider_s.pack(fill=tk.X, pady=(0, 10))
        self.slider_s.set(1.0)

        # Slider V
        ctk.CTkLabel(
            adjusts,
            text="Valor/Brilho (V): 1.0x",
            font=ctk.CTkFont(size=11),
            text_color=("#c4d0e8", "#c4d0e8"),
        ).pack(anchor="w", pady=(0, 4))

        self.slider_v = ctk.CTkSlider(
            adjusts,
            from_=0.0,
            to=2.0,
            number_of_steps=200,
            command=self._on_slider_changed,
            state="disabled",
        )
        self.slider_v.pack(fill=tk.X, pady=(0, 10))
        self.slider_v.set(1.0)

        # Botão reset
        self.btn_reset = ctk.CTkButton(
            adjusts,
            text="Resetar ajustes",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=self._reset_ajustes,
            state="disabled",
        )
        self.btn_reset.pack(fill=tk.X, pady=(4, 0))

        # Separador
        ctk.CTkFrame(ctrl_frame, fg_color="#2a3540", height=1).pack(
            fill=tk.X, padx=12, pady=12
        )

        # Seção: Filtros
        ctk.CTkLabel(
            ctrl_frame,
            text="Filtros",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(anchor="w", padx=14, pady=(0, 10))

        filtros = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        filtros.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.btn_blur_gauss = ctk.CTkButton(
            filtros,
            text="Blur Gaussiano",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=lambda: self._aplicar_filtro("blur_gauss"),
            state="disabled",
        )
        self.btn_blur_gauss.pack(fill=tk.X, padx=2, pady=(0, 6))

        self.btn_blur_med = ctk.CTkButton(
            filtros,
            text="Blur Mediano",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=lambda: self._aplicar_filtro("blur_med"),
            state="disabled",
        )
        self.btn_blur_med.pack(fill=tk.X, padx=2, pady=(0, 6))

        self.btn_canny = ctk.CTkButton(
            filtros,
            text="Bordas (Canny)",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=lambda: self._aplicar_filtro("canny"),
            state="disabled",
        )
        self.btn_canny.pack(fill=tk.X, padx=2, pady=(0, 6))

        self.btn_sobel = ctk.CTkButton(
            filtros,
            text="Bordas (Sobel)",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2d3d52", "#2d3d52"),
            hover_color=("#3d516d", "#3d516d"),
            command=lambda: self._aplicar_filtro("sobel"),
            state="disabled",
        )
        self.btn_sobel.pack(fill=tk.X, padx=2, pady=(0, 6))

        # Info
        self.lbl_info = ctk.CTkLabel(
            ctrl_frame,
            text="Nenhuma imagem carregada.",
            font=ctk.CTkFont(size=10),
            text_color=("#6b7c93", "#6b7c93"),
            justify="left",
            wraplength=250,
        )
        self.lbl_info.pack(fill=tk.X, padx=14, pady=12)

    def _build_visualizations(self, parent: ctk.CTkFrame) -> None:
        """Painel de visualizações à direita."""
        # Abas para diferentes visualizações
        self.tabview = ctk.CTkTabview(parent, fg_color=("#151c26", "#151c26"), corner_radius=12)
        self.tabview.pack(fill=tk.BOTH, expand=True)

        # Aba: Canais HSV
        self.tab_canais = self.tabview.add("Canais HSV")
        self._build_tab_canais(self.tab_canais)

        # Aba: Histogramas
        self.tab_histogramas = self.tabview.add("Histogramas")
        self._build_tab_histogramas(self.tab_histogramas)

        # Aba: Comparação
        self.tab_comparacao = self.tabview.add("Original vs Editado")
        self._build_tab_comparacao(self.tab_comparacao)

    def _build_tab_canais(self, tab: ctk.CTkFrame) -> None:
        """Visualização dos 3 canais HSV."""
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # RGB original
        col1 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ctk.CTkLabel(
            col1,
            text="RGB Original",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(pady=(10, 6))

        self.lbl_rgb = ctk.CTkLabel(
            col1, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_rgb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # HSV - Matiz
        col2 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ctk.CTkLabel(
            col2,
            text="Matiz (H)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#c4a8e0", "#c4a8e0"),
        ).pack(pady=(10, 6))

        self.lbl_hsv_h = ctk.CTkLabel(
            col2, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hsv_h.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # HSV - Saturação
        col3 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ctk.CTkLabel(
            col3,
            text="Saturação (S)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#a8e0c4", "#a8e0c4"),
        ).pack(pady=(10, 6))

        self.lbl_hsv_s = ctk.CTkLabel(
            col3, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hsv_s.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # HSV - Valor
        col4 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ctk.CTkLabel(
            col4,
            text="Valor (V)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#e0c4a8", "#e0c4a8"),
        ).pack(pady=(10, 6))

        self.lbl_hsv_v = ctk.CTkLabel(
            col4, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hsv_v.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _build_tab_histogramas(self, tab: ctk.CTkFrame) -> None:
        """Visualização de histogramas."""
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Histograma H
        col1 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ctk.CTkLabel(
            col1,
            text="Histograma Matiz (H)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#c4a8e0", "#c4a8e0"),
        ).pack(pady=(10, 6))

        self.lbl_hist_h = ctk.CTkLabel(
            col1, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hist_h.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Histograma S
        col2 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        ctk.CTkLabel(
            col2,
            text="Histograma Saturação (S)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#a8e0c4", "#a8e0c4"),
        ).pack(pady=(10, 6))

        self.lbl_hist_s = ctk.CTkLabel(
            col2, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hist_s.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Histograma V
        col3 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ctk.CTkLabel(
            col3,
            text="Histograma Valor (V)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#e0c4a8", "#e0c4a8"),
        ).pack(pady=(10, 6))

        self.lbl_hist_v = ctk.CTkLabel(
            col3, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_hist_v.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _build_tab_comparacao(self, tab: ctk.CTkFrame) -> None:
        """Comparação antes/depois."""
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        col1 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ctk.CTkLabel(
            col1,
            text="Original",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#a8c0e0", "#a8c0e0"),
        ).pack(pady=(10, 6))

        self.lbl_comp_original = ctk.CTkLabel(
            col1, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_comp_original.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        col2 = ctk.CTkFrame(container, fg_color=("#1e2733", "#1e2733"), corner_radius=10)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ctk.CTkLabel(
            col2,
            text="Editado",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#c4a8e0", "#c4a8e0"),
        ).pack(pady=(10, 6))

        self.lbl_comp_editado = ctk.CTkLabel(
            col2, text="Nenhuma imagem", fg_color=("#0d1117", "#0d1117"), corner_radius=8
        )
        self.lbl_comp_editado.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color=("#151c26", "#151c26"), corner_radius=12)
        footer.pack(fill=tk.X, padx=16, pady=(0, 16))

        self.lbl_status = ctk.CTkLabel(
            footer,
            text="Pronto.",
            font=ctk.CTkFont(size=11),
            text_color=("#6b7c93", "#6b7c93"),
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X, padx=14, pady=10)

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
        except Exception as exc:
            messagebox.showerror("Erro ao abrir imagem", str(exc), parent=self)
            return

        self._caminho = caminho
        self._rgb = rgb
        self._hsv = hsv
        self._hsv_editado = hsv.copy()

        self._atualizar_visualizacoes()
        self._habilitar_controles()

        self.lbl_info.configure(
            text=f"Arquivo: {caminho.split('/')[-1]}\nTamanho: {rgb.shape[1]}x{rgb.shape[0]} px"
        )
        self.lbl_status.configure(text="Imagem carregada com sucesso.")

    def _habilitar_controles(self) -> None:
        """Habilita todos os botões e sliders."""
        self.btn_salvar.configure(state="normal")
        self.btn_reset.configure(state="normal")
        self.slider_h.configure(state="normal")
        self.slider_s.configure(state="normal")
        self.slider_v.configure(state="normal")
        self.btn_blur_gauss.configure(state="normal")
        self.btn_blur_med.configure(state="normal")
        self.btn_canny.configure(state="normal")
        self.btn_sobel.configure(state="normal")

    def _on_slider_changed(self, value: float) -> None:
        """Callback dos sliders."""
        if self._atualizando_sliders or self._hsv is None:
            return

        self._atualizando_sliders = True

        self._ajuste_h = int(self.slider_h.get())
        self._ajuste_s = self.slider_s.get()
        self._ajuste_v = self.slider_v.get()

        # Recalcular HSV editado
        self._hsv_editado = ajustar_hsv(
            self._hsv, self._ajuste_h, self._ajuste_s, self._ajuste_v
        )

        self._atualizar_visualizacoes()
        self.lbl_status.configure(text="Ajustes aplicados.")

        self._atualizando_sliders = False

    def _reset_ajustes(self) -> None:
        """Reseta os ajustes aos valores padrão."""
        self._atualizando_sliders = True
        self.slider_h.set(0)
        self.slider_s.set(1.0)
        self.slider_v.set(1.0)
        self._atualizando_sliders = False

        self._ajuste_h = 0
        self._ajuste_s = 1.0
        self._ajuste_v = 1.0
        self._hsv_editado = self._hsv.copy()

        self._atualizar_visualizacoes()
        self.lbl_status.configure(text="Ajustes resetados.")

    def _atualizar_visualizacoes(self) -> None:
        """Atualiza todas as abas com as imagens atuais."""
        if self._rgb is None or self._hsv_editado is None:
            return

        try:
            # RGB original
            self.lbl_rgb.configure(
                image=array_para_ctk_image(self._rgb, PAINEL_MAX), text=""
            )

            # Canais HSV
            h_preview = hsv_para_preview_matiz(self._hsv_editado)
            self.lbl_hsv_h.configure(
                image=array_para_ctk_image(h_preview, PAINEL_MAX), text=""
            )

            s_preview = hsv_para_preview_saturacao(self._hsv_editado)
            self.lbl_hsv_s.configure(
                image=array_para_ctk_image(s_preview, PAINEL_MAX), text=""
            )

            v_preview = hsv_para_preview_valor(self._hsv_editado)
            self.lbl_hsv_v.configure(
                image=array_para_ctk_image(v_preview, PAINEL_MAX), text=""
            )

            # Histogramas
            hist_h = gerar_histograma(self._hsv_editado, canal=0, bins=180)
            self.lbl_hist_h.configure(
                image=array_para_ctk_image(hist_h, (320, 200)), text=""
            )

            hist_s = gerar_histograma(self._hsv_editado, canal=1, bins=256)
            self.lbl_hist_s.configure(
                image=array_para_ctk_image(hist_s, (320, 200)), text=""
            )

            hist_v = gerar_histograma(self._hsv_editado, canal=2, bins=256)
            self.lbl_hist_v.configure(
                image=array_para_ctk_image(hist_v, (320, 200)), text=""
            )

            # Comparação
            h_preview_orig = hsv_para_preview_matiz(self._hsv)
            self.lbl_comp_original.configure(
                image=array_para_ctk_image(h_preview_orig, PAINEL_MAX), text=""
            )
            self.lbl_comp_editado.configure(
                image=array_para_ctk_image(h_preview, PAINEL_MAX), text=""
            )
        except Exception as e:
            self.lbl_status.configure(text=f"Erro ao atualizar: {e}")

    def _aplicar_filtro(self, tipo: str) -> None:
        """Aplica um filtro à imagem."""
        if self._hsv_editado is None or self._rgb is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.", parent=self)
            return

        try:
            if tipo == "blur_gauss":
                self._hsv_editado = aplicar_blur_gaussiano(self._hsv_editado, kernel_size=5)
                self.lbl_status.configure(text="Filtro Blur Gaussiano aplicado.")
            elif tipo == "blur_med":
                self._hsv_editado = aplicar_blur_mediano(self._hsv_editado, kernel_size=5)
                self.lbl_status.configure(text="Filtro Blur Mediano aplicado.")
            elif tipo == "canny":
                resultado = detectar_bordas_canny(self._rgb)
                # Converter volta para HSV para manter compatibilidade
                self._hsv_editado = rgb_para_hsv(resultado)
                self.lbl_status.configure(text="Filtro Canny aplicado.")
            elif tipo == "sobel":
                resultado = detectar_bordas_sobel(self._rgb)
                self._hsv_editado = rgb_para_hsv(resultado)
                self.lbl_status.configure(text="Filtro Sobel aplicado.")

            self._atualizar_visualizacoes()
        except Exception as e:
            messagebox.showerror("Erro ao aplicar filtro", str(e), parent=self)

    def salvar_hsv(self) -> None:
        if self._hsv_editado is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem antes.", parent=self)
            return

        caminho = filedialog.asksaveasfilename(
            parent=self,
            title="Salvar imagem HSV",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("JPG", "*.jpg *.jpeg"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not caminho:
            return

        ok = cv2.imwrite(caminho, self._hsv_editado)
        if not ok:
            messagebox.showerror("Erro", "Não foi possível salvar o arquivo.", parent=self)
            return

        self.lbl_status.configure(text=f"Imagem salva em: {caminho.split('/')[-1]}")
        messagebox.showinfo("OK", f"Imagem salva com sucesso!", parent=self)


def run() -> None:
    App().mainloop()
