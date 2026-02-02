"""
Módulo de Segmentação de Íris e Pupila
======================================
Implementa detecção e segmentação da íris e pupila usando active contours.

Técnicas implementadas:
- Detecção de pupila com limiarização adaptativa e active contours
- Detecção de íris com active contours
- Refinamento de contornos com múltiplos canais HSV

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
    UFABC - Programa de Iniciação Científica
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
import warnings

from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    """
    Resultado da segmentação de íris e pupila.
    
    Attributes:
        pupil_center: Centro da pupila (x, y)
        pupil_radius: Raio da pupila
        pupil_contour: Contorno da pupila (N, 2)
        iris_center: Centro da íris (x, y)
        iris_radius: Raio da íris
        iris_contour: Contorno da íris (N, 2)
        success: Se a segmentação foi bem-sucedida
        message: Mensagem de erro ou sucesso
    """
    pupil_center: Tuple[int, int] = (0, 0)
    pupil_radius: int = 0
    pupil_contour: Optional[np.ndarray] = None
    iris_center: Tuple[int, int] = (0, 0)
    iris_radius: int = 0
    iris_contour: Optional[np.ndarray] = None
    success: bool = False
    message: str = ""
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'pupil_center': self.pupil_center,
            'pupil_radius': self.pupil_radius,
            'iris_center': self.iris_center,
            'iris_radius': self.iris_radius,
            'success': self.success,
            'message': self.message
        }


class IrisSegmenter:
    """
    Segmentador de íris e pupila usando active contours.
    
    Implementa o algoritmo de segmentação descrito no artigo,
    utilizando:
    1. Pré-processamento morfológico
    2. Limiarização adaptativa para inicialização
    3. Active contours (snakes) para refinamento
    4. Validação geométrica dos resultados
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Inicializa o segmentador.
        
        Args:
            config: Configuração de segmentação. Se None, usa padrão.
        """
        self.config = config or SegmentationConfig()
    
    def _create_circle_contour(self, center_x: float, center_y: float, 
                                radius: float, n_points: int = 400) -> np.ndarray:
        """
        Cria um contorno circular.
        
        Args:
            center_x: Coordenada x do centro
            center_y: Coordenada y do centro
            radius: Raio do círculo
            n_points: Número de pontos no contorno
            
        Returns:
            Array (n_points, 2) com coordenadas do contorno
        """
        s = np.linspace(0, 2 * np.pi, n_points)
        x = center_x + radius * np.cos(s)
        y = center_y + radius * np.sin(s)
        return np.column_stack([x, y])
    
    def _get_contour_area(self, contour: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calcula área de um contorno.
        
        Args:
            contour: Contorno (N, 2)
            
        Returns:
            Tupla (área, contorno formatado para OpenCV)
        """
        cnt = np.int32(contour).reshape(-1, 1, 2)
        area = cv2.contourArea(cnt)
        return area, cnt
    
    def _fit_circle_to_contour(self, contour: np.ndarray
                                ) -> Tuple[Tuple[int, int], int, np.ndarray]:
        """
        Ajusta um círculo a um contorno.
        
        Args:
            contour: Contorno (N, 2) ou (N, 1, 2)
            
        Returns:
            Tupla (centro, raio, contorno_circular)
        """
        if len(contour.shape) == 2:
            cnt = contour.reshape(-1, 1, 2).astype(np.float32)
        else:
            cnt = contour.astype(np.float32)
        
        center, radius = cv2.minEnclosingCircle(cnt)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        
        # Cria contorno circular
        circle_contour = self._create_circle_contour(center[0], center[1], radius)
        
        return center, radius, circle_contour
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pré-processa imagem para segmentação.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Tupla (imagem_processada, imagem_hsv)
        """
        # Redimensiona se necessário
        h, w = image.shape[:2]
        target_size = self.config.target_size
        
        if (h, w) != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Aplica blur e operações morfológicas
        processed = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Downsampling para processamento mais rápido
        for _ in range(self.config.downsample_levels):
            processed = cv2.pyrDown(processed)
        
        # Operações morfológicas para reduzir ruído
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.config.morph_kernel_size
        )
        processed = cv2.erode(processed, kernel)
        processed = cv2.dilate(processed, kernel)
        
        # Converte para HSV
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        return processed, hsv
    
    def _detect_pupil(self, hsv_image: np.ndarray, 
                       processed_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Detecta a pupila usando limiarização e active contours.
        
        Args:
            hsv_image: Imagem HSV
            processed_image: Imagem pré-processada BGR
            
        Returns:
            Tupla (contorno_pupila, centro, raio)
        """
        # Usa canal V (Value) para detecção inicial
        value_channel = hsv_image[:, :, 2]
        
        h, w = value_channel.shape
        
        # Limiarização adaptativa
        threshold = self.config.pupil_threshold_init
        _, thresh = cv2.threshold(value_channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Converte para float para active contour
        thresh_float = rgb2gray(thresh) if len(thresh.shape) == 3 else thresh.astype(float)
        
        # Inicializa contorno circular no centro da imagem
        init_radius = self.config.pupil_init_radius
        init = self._create_circle_contour(w / 2, h / 2, init_radius)
        
        # Aplica active contour
        snake = active_contour(
            gaussian(thresh_float, 3), 
            init,
            alpha=self.config.snake_alpha,
            beta=self.config.snake_beta
        )
        
        area, cnt = self._get_contour_area(snake)
        
        # Ajusta threshold se área muito pequena
        attempts = 0
        while area < self.config.min_pupil_area and attempts < 10:
            threshold += 10
            _, thresh = cv2.threshold(value_channel, threshold, 255, cv2.THRESH_BINARY_INV)
            thresh_float = thresh.astype(float)
            
            snake = active_contour(
                gaussian(thresh_float, 3),
                init,
                alpha=self.config.snake_alpha,
                beta=self.config.snake_beta
            )
            area, cnt = self._get_contour_area(snake)
            attempts += 1
        
        # Refinamento com canal S (Saturation)
        sat_channel = hsv_image[:, :, 1]
        snake_refined = active_contour(
            gaussian(sat_channel, 3),
            snake,
            alpha=self.config.snake_alpha,
            beta=self.config.snake_beta
        )
        
        area_refined, cnt_refined = self._get_contour_area(snake_refined)
        
        # Usa contorno refinado se área razoável
        if area_refined > 0 and not (area > 5 * area_refined):
            snake = snake_refined
            cnt = cnt_refined
            area = area_refined
        
        # Calcula centro e raio
        center, radius, _ = self._fit_circle_to_contour(snake)
        
        return snake, center, radius
    
    def _detect_iris(self, hsv_image: np.ndarray,
                      pupil_contour: np.ndarray,
                      pupil_center: Tuple[int, int],
                      pupil_radius: int) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Detecta a íris usando active contours expandindo da pupila.
        
        Args:
            hsv_image: Imagem HSV
            pupil_contour: Contorno da pupila
            pupil_center: Centro da pupila
            pupil_radius: Raio da pupila
            
        Returns:
            Tupla (contorno_iris, centro, raio)
        """
        value_channel = hsv_image[:, :, 2]
        value_channel = cv2.medianBlur(value_channel, 5)
        
        h, w = value_channel.shape
        
        # Inicializa contorno maior que a pupila
        if pupil_radius < 40:
            init_radius = 2 * pupil_radius
        else:
            init_radius = 1.4 * pupil_radius
        
        init = self._create_circle_contour(pupil_center[0], pupil_center[1], init_radius)
        
        # Active contour com parâmetros para expansão
        convergence = self.config.iris_convergence
        pupil_area, _ = self._get_contour_area(pupil_contour)
        iris_area = 10**20
        
        attempts = 0
        while iris_area > 10 * pupil_area and attempts < 10:
            snake = active_contour(
                gaussian(value_channel, 3),
                init,
                alpha=self.config.iris_snake_alpha,
                beta=self.config.snake_beta,
                gamma=self.config.iris_snake_gamma,
                convergence=convergence
            )
            iris_area, cnt = self._get_contour_area(snake)
            convergence *= 1.2
            attempts += 1
        
        # Ajusta círculo ao contorno
        center, radius, circle_contour = self._fit_circle_to_contour(snake)
        
        # Verifica se contorno excede limites da imagem
        if center[1] + radius > h:
            # Recalcula usando distância mínima do centro ao contorno
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center = (cx, cy)
                
                # Calcula raio como distância mínima
                distances = [distance.euclidean(center, (snake[i, 0], snake[i, 1])) 
                            for i in range(len(snake))]
                radius = int(min(distances))
                
                circle_contour = self._create_circle_contour(cx, cy, radius)
        
        # Ajusta se ainda excede
        if center[1] + radius > h:
            diff = center[1] + radius - h + 1
            radius = radius - diff
            circle_contour = self._create_circle_contour(center[0], center[1], radius)
        
        return circle_contour, center, radius
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Realiza segmentação completa de íris e pupila.
        
        Args:
            image: Imagem BGR do olho
            
        Returns:
            SegmentationResult com contornos e parâmetros
        """
        result = SegmentationResult()
        
        try:
            # Pré-processamento
            processed, hsv = self._preprocess_image(image)
            
            # Detecta pupila
            pupil_contour, pupil_center, pupil_radius = self._detect_pupil(hsv, processed)
            
            if pupil_radius < self.config.min_pupil_radius:
                result.message = f"Pupila muito pequena: raio={pupil_radius}"
                return result
            
            # Detecta íris
            iris_contour, iris_center, iris_radius = self._detect_iris(
                hsv, pupil_contour, pupil_center, pupil_radius
            )
            
            if iris_radius < pupil_radius:
                result.message = f"Íris menor que pupila: iris={iris_radius}, pupila={pupil_radius}"
                return result
            
            # Preenche resultado
            result.pupil_center = pupil_center
            result.pupil_radius = pupil_radius
            result.pupil_contour = pupil_contour
            result.iris_center = iris_center
            result.iris_radius = iris_radius
            result.iris_contour = iris_contour
            result.success = True
            result.message = "Segmentação bem-sucedida"
            
        except Exception as e:
            result.message = f"Erro na segmentação: {str(e)}"
        
        return result
    
    def segment_batch(self, images: np.ndarray, 
                       verbose: bool = True) -> List[SegmentationResult]:
        """
        Segmenta múltiplas imagens.
        
        Args:
            images: Array de imagens (N, H, W, C)
            verbose: Se True, mostra progresso
            
        Returns:
            Lista de SegmentationResult
        """
        results = []
        n_images = len(images)
        
        for i, img in enumerate(images):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Segmentando imagem {i + 1}/{n_images}...")
            
            result = self.segment(img)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        if verbose:
            print(f"  Segmentação concluída: {success_count}/{n_images} bem-sucedidas")
        
        return results
    
    def visualize(self, image: np.ndarray, 
                  result: SegmentationResult,
                  show: bool = True) -> np.ndarray:
        """
        Visualiza resultado da segmentação.
        
        Args:
            image: Imagem original
            result: Resultado da segmentação
            show: Se True, exibe a imagem
            
        Returns:
            Imagem com contornos desenhados
        """
        import matplotlib.pyplot as plt
        
        vis_image = image.copy()
        
        if result.success:
            # Desenha contorno da pupila (verde)
            if result.pupil_contour is not None:
                pts = result.pupil_contour.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
            
            # Desenha contorno da íris (vermelho)
            if result.iris_contour is not None:
                pts = result.iris_contour.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (0, 0, 255), 2)
            
            # Desenha centros
            cv2.circle(vis_image, result.pupil_center, 3, (0, 255, 0), -1)
            cv2.circle(vis_image, result.iris_center, 3, (0, 0, 255), -1)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Segmentação: {result.message}")
            plt.axis('off')
            plt.show()
        
        return vis_image


def create_segmenter(config: Optional[SegmentationConfig] = None) -> IrisSegmenter:
    """
    Factory function para criar segmentador.
    
    Args:
        config: Configuração de segmentação
        
    Returns:
        IrisSegmenter configurado
    """
    return IrisSegmenter(config)
