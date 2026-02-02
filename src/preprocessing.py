"""
Módulo de Pré-processamento de Imagens
======================================
Implementa transformações fotométricas e normalização de imagens de íris.

Transformações implementadas conforme metodologia do artigo:
- Equalização global de histograma
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from enum import Enum

from .config import PreprocessingConfig, PhotometricTransform, ColorChannel


class ImagePreprocessor:
    """
    Classe para pré-processamento de imagens de íris.
    
    Implementa transformações fotométricas padronizadas para teste de robustez
    do pipeline conforme descrito na metodologia do artigo.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializa o pré-processador.
        
        Args:
            config: Configuração de pré-processamento. Se None, usa padrão.
        """
        self.config = config or PreprocessingConfig()
        
        # Inicializa CLAHE
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )
    
    def extract_channel(self, image: np.ndarray, 
                        channel: Optional[ColorChannel] = None) -> np.ndarray:
        """
        Extrai um canal específico da imagem.
        
        Args:
            image: Imagem BGR (OpenCV format)
            channel: Canal a extrair. Se None, usa o configurado.
            
        Returns:
            Imagem em escala de cinza do canal selecionado.
        """
        channel = channel or self.config.color_channel
        
        if channel == ColorChannel.GRAY:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        
        elif channel in [ColorChannel.H, ColorChannel.S, ColorChannel.V]:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_idx = {'h': 0, 's': 1, 'v': 2}[channel.value]
            return hsv[:, :, channel_idx]
        
        elif channel in [ColorChannel.B, ColorChannel.G, ColorChannel.R]:
            channel_idx = {'b': 0, 'g': 1, 'r': 2}[channel.value]
            return image[:, :, channel_idx]
        
        else:
            raise ValueError(f"Canal não suportado: {channel}")
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica equalização global de histograma.
        
        Esta transformação redistribui os valores de intensidade para
        ocupar todo o range disponível, aumentando o contraste global.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Imagem equalizada
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(image)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE divide a imagem em tiles e aplica equalização adaptativa
        em cada um, com limite de contraste para evitar amplificação
        excessiva de ruído.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Imagem com CLAHE aplicado
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(image)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica Gaussian blur (suavização gaussiana).
        
        Útil para reduzir ruído de alta frequência e avaliar
        se o modelo depende de detalhes finos ou padrões mais grosseiros.
        
        Args:
            image: Imagem (pode ser colorida ou escala de cinza)
            
        Returns:
            Imagem suavizada
        """
        return cv2.GaussianBlur(
            image, 
            self.config.blur_kernel_size, 
            self.config.blur_sigma
        )
    
    def apply_transform(self, image: np.ndarray, 
                        transform: PhotometricTransform) -> np.ndarray:
        """
        Aplica uma transformação fotométrica específica.
        
        Args:
            image: Imagem de entrada
            transform: Tipo de transformação a aplicar
            
        Returns:
            Imagem transformada
        """
        if transform == PhotometricTransform.ORIGINAL:
            return image.copy()
        
        elif transform == PhotometricTransform.HISTOGRAM:
            return self.apply_histogram_equalization(image)
        
        elif transform == PhotometricTransform.CLAHE:
            return self.apply_clahe(image)
        
        elif transform == PhotometricTransform.BLUR:
            return self.apply_gaussian_blur(image)
        
        else:
            raise ValueError(f"Transformação não suportada: {transform}")
    
    def preprocess(self, image: np.ndarray, 
                   channel: Optional[ColorChannel] = None,
                   transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                   ) -> np.ndarray:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            image: Imagem BGR de entrada
            channel: Canal de cor a extrair
            transform: Transformação fotométrica a aplicar
            
        Returns:
            Imagem pré-processada em escala de cinza
        """
        # Extrai canal
        gray = self.extract_channel(image, channel)
        
        # Aplica transformação fotométrica
        processed = self.apply_transform(gray, transform)
        
        return processed
    
    def preprocess_batch(self, images: np.ndarray,
                         channel: Optional[ColorChannel] = None,
                         transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                         ) -> np.ndarray:
        """
        Pré-processa um lote de imagens.
        
        Args:
            images: Array de imagens (N, H, W, C) ou (N, H, W)
            channel: Canal de cor a extrair
            transform: Transformação fotométrica a aplicar
            
        Returns:
            Array de imagens pré-processadas (N, H, W)
        """
        n_images = images.shape[0]
        
        # Processa primeira imagem para determinar shape
        first_processed = self.preprocess(
            np.squeeze(images[0]), channel, transform
        )
        
        # Cria array de saída
        output = np.zeros((n_images,) + first_processed.shape, dtype=np.uint8)
        output[0] = first_processed
        
        # Processa restante
        for i in range(1, n_images):
            output[i] = self.preprocess(
                np.squeeze(images[i]), channel, transform
            )
        
        return output


class IrisNormalizer:
    """
    Normalização geométrica da íris (rubber sheet transform).
    
    Mapeia a região anular da íris para uma representação retangular
    em coordenadas polares, permitindo comparação ponto-a-ponto
    mesmo com variações de dilatação pupilar.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializa o normalizador.
        
        Args:
            config: Configuração de pré-processamento
        """
        self.config = config or PreprocessingConfig()
        self.height = self.config.normalized_height
        self.width = self.config.normalized_width
    
    def normalize(self, image: np.ndarray,
                  pupil_center: Tuple[int, int],
                  pupil_contour: np.ndarray,
                  iris_contour: np.ndarray) -> np.ndarray:
        """
        Aplica normalização rubber sheet.
        
        Mapeia a região anular entre pupila e íris para uma grade retangular.
        
        Args:
            image: Imagem de entrada (BGR ou grayscale)
            pupil_center: Centro da pupila (x, y)
            pupil_contour: Contorno da pupila
            iris_contour: Contorno da íris
            
        Returns:
            Imagem normalizada (height x width)
        """
        # Garante que a imagem tem 3 canais
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Cria imagem de saída
        normalized = np.zeros(
            (self.height, self.width, image.shape[2]), 
            dtype=np.uint8
        )
        
        # Varre todos os ângulos
        for col, theta_deg in enumerate(np.linspace(0, 360, self.width, endpoint=False)):
            theta_rad = np.deg2rad(theta_deg)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            
            # Encontra ponto na borda da pupila
            px, py = float(pupil_center[0]), float(pupil_center[1])
            while cv2.pointPolygonTest(pupil_contour, (int(px), int(py)), False) == 1:
                px += cos_t
                py += sin_t
            pupil_edge = (int(px - cos_t), int(py - sin_t))
            
            # Encontra ponto na borda da íris
            ix, iy = float(pupil_center[0]), float(pupil_center[1])
            while cv2.pointPolygonTest(iris_contour, (int(ix), int(iy)), False) == 1:
                ix += cos_t
                iy += sin_t
            iris_edge = (int(ix - cos_t), int(iy - sin_t))
            
            # Interpola entre pupila e íris
            for row, r in enumerate(np.linspace(0, 1, self.height)):
                x = int((1 - r) * pupil_edge[0] + r * iris_edge[0])
                y = int((1 - r) * pupil_edge[1] + r * iris_edge[1])
                
                # Verifica limites
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized[row, col] = image[y, x]
        
        return normalized
    
    def crop_upper_region(self, normalized_image: np.ndarray) -> np.ndarray:
        """
        Remove região superior (pálpebra) da imagem normalizada.
        
        Args:
            normalized_image: Imagem normalizada
            
        Returns:
            Imagem com região superior removida
        """
        return normalized_image[self.config.upper_cut_height:, :]


def create_preprocessor(config: Optional[PreprocessingConfig] = None) -> ImagePreprocessor:
    """Factory function para criar pré-processador."""
    return ImagePreprocessor(config)


def create_normalizer(config: Optional[PreprocessingConfig] = None) -> IrisNormalizer:
    """Factory function para criar normalizador."""
    return IrisNormalizer(config)


# =============================================================================
# Funções utilitárias
# =============================================================================

def apply_all_transforms(image: np.ndarray, 
                         config: Optional[PreprocessingConfig] = None
                         ) -> dict:
    """
    Aplica todas as transformações fotométricas a uma imagem.
    
    Útil para análise comparativa de robustez.
    
    Args:
        image: Imagem de entrada
        config: Configuração de pré-processamento
        
    Returns:
        Dicionário com imagens transformadas
    """
    preprocessor = ImagePreprocessor(config)
    
    results = {}
    for transform in PhotometricTransform:
        results[transform.value] = preprocessor.apply_transform(image, transform)
    
    return results


def normalize_intensity(image: np.ndarray, 
                        target_mean: float = 128.0,
                        target_std: float = 50.0) -> np.ndarray:
    """
    Normaliza intensidade da imagem para média e desvio padrão alvo.
    
    Args:
        image: Imagem de entrada
        target_mean: Média alvo
        target_std: Desvio padrão alvo
        
    Returns:
        Imagem normalizada
    """
    current_mean = np.mean(image)
    current_std = np.std(image)
    
    if current_std == 0:
        return np.full_like(image, target_mean, dtype=np.uint8)
    
    normalized = (image - current_mean) / current_std * target_std + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)
