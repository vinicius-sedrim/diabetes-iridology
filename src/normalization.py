"""
Módulo de Normalização de Íris (Rubber Sheet)
=============================================
Implementa transformação de coordenadas polares para retangulares.

Técnica implementada:
- Rubber sheet model (Daugman): mapeia a região anelar da íris
  (entre pupila e limbo) para uma representação retangular normalizada.

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
    UFABC - Programa de Iniciação Científica
    
    Daugman, J. "How iris recognition works." 
    IEEE Trans. Circuits and Systems for Video Technology, 2004.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from .config import NormalizationConfig
from .segmentation import SegmentationResult


@dataclass
class NormalizationResult:
    """
    Resultado da normalização de íris.
    
    Attributes:
        normalized_image: Imagem normalizada (H, W) ou (H, W, C)
        angular_resolution: Resolução angular (graus por pixel)
        radial_resolution: Resolução radial (fração por pixel)
        success: Se a normalização foi bem-sucedida
        message: Mensagem de erro ou sucesso
    """
    normalized_image: Optional[np.ndarray] = None
    angular_resolution: float = 0.0
    radial_resolution: float = 0.0
    success: bool = False
    message: str = ""


class IrisNormalizer:
    """
    Normalizador de íris usando transformação rubber sheet.
    
    Mapeia a região anelar da íris (entre pupila e limbo esclerocorneano)
    para uma representação retangular de dimensões fixas, permitindo
    comparação ponto-a-ponto mesmo com variações de:
    - Dilatação pupilar
    - Rotação do olho
    - Enquadramento da câmera
    
    A transformação segue o modelo de Daugman:
    I(x(r,θ), y(r,θ)) → I(r, θ)
    
    Onde:
    - r ∈ [0, 1] representa a posição radial normalizada
    - θ ∈ [0, 2π] representa o ângulo
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Inicializa o normalizador.
        
        Args:
            config: Configuração de normalização. Se None, usa padrão.
        """
        self.config = config or NormalizationConfig()
    
    def _interpolate_point(self, image: np.ndarray, 
                           x: float, y: float) -> np.ndarray:
        """
        Interpola valor de pixel em coordenadas fracionárias.
        
        Usa interpolação bilinear para obter valores suaves
        em posições não-inteiras.
        
        Args:
            image: Imagem de entrada
            x: Coordenada x (pode ser fracionária)
            y: Coordenada y (pode ser fracionária)
            
        Returns:
            Valor do pixel (escalar ou vetor RGB)
        """
        h, w = image.shape[:2]
        
        # Limita coordenadas aos limites da imagem
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Pesos para interpolação bilinear
        wx = x - x0
        wy = y - y0
        
        if len(image.shape) == 3:
            # Imagem colorida
            val = (1 - wx) * (1 - wy) * image[y0, x0] + \
                  wx * (1 - wy) * image[y0, x1] + \
                  (1 - wx) * wy * image[y1, x0] + \
                  wx * wy * image[y1, x1]
        else:
            # Imagem em escala de cinza
            val = (1 - wx) * (1 - wy) * image[y0, x0] + \
                  wx * (1 - wy) * image[y0, x1] + \
                  (1 - wx) * wy * image[y1, x0] + \
                  wx * wy * image[y1, x1]
        
        return val
    
    def normalize_from_contours(self, 
                                 image: np.ndarray,
                                 pupil_center: Tuple[float, float],
                                 pupil_contour: np.ndarray,
                                 iris_contour: np.ndarray) -> NormalizationResult:
        """
        Normaliza íris usando contornos (não necessariamente circulares).
        
        Esta versão usa os contornos reais da pupila e íris para
        uma normalização mais precisa que considera variações
        na forma dos contornos.
        
        Args:
            image: Imagem original (BGR ou grayscale)
            pupil_center: Centro da pupila (x, y)
            pupil_contour: Contorno da pupila formatado para cv2
            iris_contour: Contorno da íris formatado para cv2
            
        Returns:
            NormalizationResult com imagem normalizada
        """
        result = NormalizationResult()
        
        try:
            height = self.config.normalized_height
            width = self.config.normalized_width
            
            # Prepara contornos para pointPolygonTest
            if len(pupil_contour.shape) == 2:
                pupil_cnt = pupil_contour.reshape(1, -1, 2).astype(np.int64)
            else:
                pupil_cnt = pupil_contour.astype(np.int64)
            
            if len(iris_contour.shape) == 2:
                iris_cnt = iris_contour.reshape(1, -1, 2).astype(np.int64)
            else:
                iris_cnt = iris_contour.astype(np.int64)
            
            # Determina tipo de saída
            if len(image.shape) == 3:
                polar_img = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
            else:
                polar_img = np.zeros((height, width), dtype=np.uint8)
            
            # Varre cada ângulo e posição radial
            for col in range(width):
                theta = col * 360.0 / width  # Ângulo em graus
                theta_rad = theta * np.pi / 180.0
                
                # Encontra ponto na borda da pupila
                px, py = float(pupil_center[0]), float(pupil_center[1])
                
                # Move do centro até sair da pupila
                while cv2.pointPolygonTest(pupil_cnt, (int(px), int(py)), False) >= 0:
                    px += np.cos(theta_rad)
                    py += np.sin(theta_rad)
                
                # Ponto na borda da pupila (recua um passo)
                pupil_x = px - np.cos(theta_rad)
                pupil_y = py - np.sin(theta_rad)
                
                # Continua até sair da íris
                while cv2.pointPolygonTest(iris_cnt, (int(px), int(py)), False) >= 0:
                    px += np.cos(theta_rad)
                    py += np.sin(theta_rad)
                
                # Ponto na borda da íris
                iris_x = px - np.cos(theta_rad)
                iris_y = py - np.sin(theta_rad)
                
                # Interpola ao longo do raio
                for row in range(height):
                    r = row / (height - 1.0)  # r ∈ [0, 1]
                    
                    # Interpolação linear entre pupila e íris
                    x = (1.0 - r) * pupil_x + r * iris_x
                    y = (1.0 - r) * pupil_y + r * iris_y
                    
                    # Obtém valor do pixel
                    polar_img[row, col] = self._interpolate_point(image, x, y)
            
            result.normalized_image = polar_img
            result.angular_resolution = 360.0 / width
            result.radial_resolution = 1.0 / height
            result.success = True
            result.message = "Normalização bem-sucedida"
            
        except Exception as e:
            result.message = f"Erro na normalização: {str(e)}"
        
        return result
    
    def normalize_from_circles(self,
                               image: np.ndarray,
                               pupil_center: Tuple[int, int],
                               pupil_radius: int,
                               iris_center: Tuple[int, int],
                               iris_radius: int) -> NormalizationResult:
        """
        Normaliza íris usando aproximação circular.
        
        Versão simplificada que assume pupila e íris circulares.
        Mais rápida que a versão com contornos.
        
        Args:
            image: Imagem original
            pupil_center: Centro da pupila (x, y)
            pupil_radius: Raio da pupila
            iris_center: Centro da íris (x, y)  
            iris_radius: Raio da íris
            
        Returns:
            NormalizationResult com imagem normalizada
        """
        result = NormalizationResult()
        
        try:
            height = self.config.normalized_height
            width = self.config.normalized_width
            
            if len(image.shape) == 3:
                polar_img = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
            else:
                polar_img = np.zeros((height, width), dtype=np.uint8)
            
            # Pré-calcula ângulos
            thetas = np.linspace(0, 2 * np.pi, width, endpoint=False)
            
            for col, theta in enumerate(thetas):
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                # Pontos nas bordas (usando centros possivelmente diferentes)
                pupil_x = pupil_center[0] + pupil_radius * cos_theta
                pupil_y = pupil_center[1] + pupil_radius * sin_theta
                
                iris_x = iris_center[0] + iris_radius * cos_theta
                iris_y = iris_center[1] + iris_radius * sin_theta
                
                for row in range(height):
                    r = row / (height - 1.0) if height > 1 else 0.0
                    
                    x = (1.0 - r) * pupil_x + r * iris_x
                    y = (1.0 - r) * pupil_y + r * iris_y
                    
                    polar_img[row, col] = self._interpolate_point(image, x, y)
            
            result.normalized_image = polar_img
            result.angular_resolution = 360.0 / width
            result.radial_resolution = 1.0 / height
            result.success = True
            result.message = "Normalização bem-sucedida"
            
        except Exception as e:
            result.message = f"Erro na normalização: {str(e)}"
        
        return result
    
    def normalize(self, image: np.ndarray,
                  segmentation: SegmentationResult,
                  use_contours: bool = True) -> NormalizationResult:
        """
        Normaliza íris a partir de resultado de segmentação.
        
        Args:
            image: Imagem original
            segmentation: Resultado da segmentação
            use_contours: Se True, usa contornos; senão, usa círculos
            
        Returns:
            NormalizationResult com imagem normalizada
        """
        if not segmentation.success:
            return NormalizationResult(
                success=False,
                message=f"Segmentação inválida: {segmentation.message}"
            )
        
        if use_contours and segmentation.pupil_contour is not None:
            return self.normalize_from_contours(
                image,
                segmentation.pupil_center,
                segmentation.pupil_contour,
                segmentation.iris_contour
            )
        else:
            return self.normalize_from_circles(
                image,
                segmentation.pupil_center,
                segmentation.pupil_radius,
                segmentation.iris_center,
                segmentation.iris_radius
            )
    
    def normalize_batch(self, images: np.ndarray,
                        segmentations: list,
                        use_contours: bool = True,
                        verbose: bool = True) -> list:
        """
        Normaliza múltiplas imagens.
        
        Args:
            images: Array de imagens (N, H, W, C) ou (N, H, W)
            segmentations: Lista de SegmentationResult
            use_contours: Se True, usa contornos
            verbose: Se True, mostra progresso
            
        Returns:
            Lista de NormalizationResult
        """
        results = []
        n_images = len(images)
        
        for i, (img, seg) in enumerate(zip(images, segmentations)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Normalizando imagem {i + 1}/{n_images}...")
            
            result = self.normalize(img, seg, use_contours)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        if verbose:
            print(f"  Normalização concluída: {success_count}/{n_images} bem-sucedidas")
        
        return results
    
    def get_normalized_images_array(self, results: list) -> np.ndarray:
        """
        Converte lista de resultados em array numpy.
        
        Args:
            results: Lista de NormalizationResult
            
        Returns:
            Array (N, H, W) ou (N, H, W, C) com imagens normalizadas
        """
        valid_images = [r.normalized_image for r in results 
                       if r.success and r.normalized_image is not None]
        
        if not valid_images:
            return np.array([])
        
        return np.array(valid_images)


class IrisPreprocessingPipeline:
    """
    Pipeline completo de pré-processamento de íris.
    
    Combina segmentação e normalização em um fluxo único.
    """
    
    def __init__(self, 
                 segmentation_config=None,
                 normalization_config=None):
        """
        Inicializa o pipeline.
        
        Args:
            segmentation_config: Configuração de segmentação
            normalization_config: Configuração de normalização
        """
        from .segmentation import IrisSegmenter
        
        self.segmenter = IrisSegmenter(segmentation_config)
        self.normalizer = IrisNormalizer(normalization_config)
    
    def process_single(self, image: np.ndarray,
                       use_contours: bool = True) -> Tuple[SegmentationResult, NormalizationResult]:
        """
        Processa uma única imagem.
        
        Args:
            image: Imagem do olho (BGR)
            use_contours: Se True, usa contornos na normalização
            
        Returns:
            Tupla (SegmentationResult, NormalizationResult)
        """
        seg_result = self.segmenter.segment(image)
        
        if not seg_result.success:
            norm_result = NormalizationResult(
                success=False,
                message=f"Segmentação falhou: {seg_result.message}"
            )
            return seg_result, norm_result
        
        norm_result = self.normalizer.normalize(image, seg_result, use_contours)
        
        return seg_result, norm_result
    
    def process_batch(self, images: np.ndarray,
                      use_contours: bool = True,
                      verbose: bool = True) -> Tuple[list, list, np.ndarray]:
        """
        Processa múltiplas imagens.
        
        Args:
            images: Array de imagens (N, H, W, C)
            use_contours: Se True, usa contornos
            verbose: Se True, mostra progresso
            
        Returns:
            Tupla (seg_results, norm_results, normalized_array)
        """
        if verbose:
            print("Etapa 1/2: Segmentação...")
        
        seg_results = self.segmenter.segment_batch(images, verbose)
        
        if verbose:
            print("Etapa 2/2: Normalização...")
        
        norm_results = self.normalizer.normalize_batch(
            images, seg_results, use_contours, verbose
        )
        
        # Extrai imagens normalizadas bem-sucedidas
        normalized_array = self.normalizer.get_normalized_images_array(norm_results)
        
        return seg_results, norm_results, normalized_array


def create_normalizer(config: Optional[NormalizationConfig] = None) -> IrisNormalizer:
    """
    Factory function para criar normalizador.
    
    Args:
        config: Configuração de normalização
        
    Returns:
        IrisNormalizer configurado
    """
    return IrisNormalizer(config)


def create_preprocessing_pipeline(segmentation_config=None,
                                   normalization_config=None) -> IrisPreprocessingPipeline:
    """
    Factory function para criar pipeline de pré-processamento.
    
    Args:
        segmentation_config: Configuração de segmentação
        normalization_config: Configuração de normalização
        
    Returns:
        IrisPreprocessingPipeline configurado
    """
    return IrisPreprocessingPipeline(segmentation_config, normalization_config)
