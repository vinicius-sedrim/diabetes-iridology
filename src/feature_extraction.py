"""
Módulo de Extração de Features
==============================
Extração de representações (features) das imagens de íris.

Features implementadas conforme metodologia do artigo:
- Pixel features (baseline)
- LBP (Local Binary Patterns)
- GLCM (Grey-Level Co-occurrence Matrix) / Haralick
- Estatísticas de intensidade (média, desvio padrão, assimetria, curtose)
- Histogramas de intensidade
- Gabor features
- HOG (Histogram of Oriented Gradients)

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Union
from scipy import ndimage as ndi
from scipy import stats
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor_kernel
from skimage.transform import rescale
import warnings

from .config import FeatureConfig, FeatureType

# Tenta importar skimage.feature.graycomatrix (versão nova) ou greycomatrix (versão antiga)
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops


class FeatureExtractor:
    """
    Extrator de features para imagens de íris normalizadas.
    
    Suporta múltiplos tipos de features que podem ser combinados
    para criar vetores de representação robustos.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Inicializa o extrator de features.
        
        Args:
            config: Configuração de extração. Se None, usa padrão.
        """
        self.config = config or FeatureConfig()
        self._gabor_kernels = None  # Cache para kernels Gabor
    
    # =========================================================================
    # Pixel Features (Baseline)
    # =========================================================================
    
    def extract_pixel_features(self, image: np.ndarray, 
                                region: Optional[Tuple[int, int]] = None
                                ) -> np.ndarray:
        """
        Extrai features baseadas em pixels (baseline).
        
        Redimensiona a imagem e achata em um vetor de intensidades.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col). Se None, usa imagem toda.
            
        Returns:
            Vetor de features de pixel
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        # Downsampling
        scale = 1.0 / self.config.pixel_downsample_size
        downsampled = rescale(image, scale, preserve_range=True, anti_aliasing=True)
        
        return downsampled.flatten().astype(np.float32)
    
    # =========================================================================
    # LBP Features
    # =========================================================================
    
    def extract_lbp_features(self, image: np.ndarray,
                              region: Optional[Tuple[int, int]] = None
                              ) -> np.ndarray:
        """
        Extrai features LBP (Local Binary Patterns).
        
        LBP captura microtexturas locais comparando cada pixel
        com seus vizinhos em um padrão circular.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col)
            
        Returns:
            Histograma de padrões LBP
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        # Calcula LBP
        lbp = local_binary_pattern(
            image,
            P=self.config.lbp_n_points,
            R=self.config.lbp_radius,
            method=self.config.lbp_method
        )
        
        # Calcula histograma
        n_bins = self.config.lbp_n_points + 2  # Para método 'uniform'
        hist, _ = np.histogram(
            lbp.ravel(), 
            bins=n_bins, 
            range=(0, n_bins),
            density=True
        )
        
        return hist.astype(np.float32)
    
    # =========================================================================
    # GLCM / Haralick Features
    # =========================================================================
    
    def extract_glcm_features(self, image: np.ndarray,
                               region: Optional[Tuple[int, int]] = None
                               ) -> np.ndarray:
        """
        Extrai features GLCM (Grey-Level Co-occurrence Matrix).
        
        GLCM captura relações espaciais entre níveis de cinza,
        permitindo caracterizar texturas por propriedades estatísticas
        (Haralick features).
        
        Args:
            image: Imagem em escala de cinza (uint8)
            region: Região angular (start_col, end_col)
            
        Returns:
            Vetor de propriedades GLCM (Haralick)
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        # Garante que imagem está em uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Quantiza para reduzir níveis (melhora estabilidade)
        levels = min(self.config.glcm_levels, 64)
        image_quantized = (image // (256 // levels)).astype(np.uint8)
        
        # Calcula GLCM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcm = graycomatrix(
                image_quantized,
                distances=self.config.glcm_distances,
                angles=self.config.glcm_angles,
                levels=levels,
                symmetric=True,
                normed=True
            )
        
        # Extrai propriedades
        features = []
        for prop in self.config.glcm_properties:
            try:
                prop_values = graycoprops(glcm, prop)
                features.extend(prop_values.flatten())
            except Exception:
                # Algumas propriedades podem falhar em casos especiais
                n_expected = len(self.config.glcm_distances) * len(self.config.glcm_angles)
                features.extend([0.0] * n_expected)
        
        return np.array(features, dtype=np.float32)
    
    # =========================================================================
    # Intensity Statistics
    # =========================================================================
    
    def extract_intensity_stats(self, image: np.ndarray,
                                 region: Optional[Tuple[int, int]] = None
                                 ) -> np.ndarray:
        """
        Extrai estatísticas de intensidade.
        
        Inclui: média, desvio padrão, assimetria (skewness),
        curtose (kurtosis) e histograma.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col)
            
        Returns:
            Vetor de estatísticas
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        pixels = image.flatten().astype(np.float64)
        
        # Estatísticas básicas
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        
        # Momentos de ordem superior
        if std_val > 0:
            skewness = stats.skew(pixels)
            kurtosis = stats.kurtosis(pixels)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Histograma normalizado
        hist, _ = np.histogram(
            pixels, 
            bins=self.config.histogram_bins,
            range=(0, 255),
            density=True
        )
        
        # Combina todas as estatísticas
        features = np.concatenate([
            [mean_val, std_val, skewness, kurtosis],
            hist
        ])
        
        return features.astype(np.float32)
    
    # =========================================================================
    # Gabor Features
    # =========================================================================
    
    def _get_gabor_kernels(self) -> List:
        """
        Gera ou retorna kernels Gabor em cache.
        
        Returns:
            Lista de kernels Gabor
        """
        if self._gabor_kernels is None:
            self._gabor_kernels = []
            n_orientations = self.config.gabor_n_orientations
            
            for theta_idx in range(n_orientations):
                theta = theta_idx / n_orientations * np.pi
                for frequency in self.config.gabor_frequencies:
                    kernel = np.real(gabor_kernel(
                        frequency,
                        theta=theta,
                        sigma_x=self.config.gabor_sigma,
                        sigma_y=self.config.gabor_sigma
                    ))
                    self._gabor_kernels.append(kernel)
        
        return self._gabor_kernels
    
    def extract_gabor_features(self, image: np.ndarray,
                                region: Optional[Tuple[int, int]] = None
                                ) -> np.ndarray:
        """
        Extrai features de filtros de Gabor.
        
        Filtros de Gabor capturam informação de frequência e
        orientação em diferentes escalas.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col)
            
        Returns:
            Vetor de features Gabor (energia média por filtro)
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        # Normaliza imagem
        image = image.astype(np.float64)
        if image.max() > 1:
            image = image / 255.0
        
        kernels = self._get_gabor_kernels()
        features = []
        
        for kernel in kernels:
            filtered = ndi.convolve(image, kernel, mode='wrap')
            # Energia média (L2)
            energy = np.mean(filtered ** 2)
            features.append(energy)
        
        return np.array(features, dtype=np.float32)
    
    # =========================================================================
    # HOG Features
    # =========================================================================
    
    def extract_hog_features(self, image: np.ndarray,
                              region: Optional[Tuple[int, int]] = None
                              ) -> np.ndarray:
        """
        Extrai features HOG (Histogram of Oriented Gradients).
        
        HOG captura informação de forma e estrutura através
        de histogramas locais de direção de gradientes.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col)
            
        Returns:
            Vetor de features HOG
        """
        if region is not None:
            image = image[:, region[0]:region[1]]
        
        # Garante dimensões mínimas
        min_size = self.config.hog_pixels_per_cell[0] * 2
        if image.shape[0] < min_size or image.shape[1] < min_size:
            # Redimensiona se necessário
            scale = max(min_size / image.shape[0], min_size / image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        features = hog(
            image,
            orientations=self.config.hog_orientations,
            pixels_per_cell=self.config.hog_pixels_per_cell,
            cells_per_block=self.config.hog_cells_per_block,
            feature_vector=True
        )
        
        return features.astype(np.float32)
    
    # =========================================================================
    # Extração Combinada
    # =========================================================================
    
    def extract_features(self, image: np.ndarray,
                          region: Optional[Tuple[int, int]] = None,
                          feature_types: Optional[List[FeatureType]] = None
                          ) -> np.ndarray:
        """
        Extrai combinação de features conforme configuração.
        
        Args:
            image: Imagem em escala de cinza
            region: Região angular (start_col, end_col)
            feature_types: Lista de tipos de features. Se None, usa config.
            
        Returns:
            Vetor concatenado de todas as features
        """
        feature_types = feature_types or self.config.features
        
        all_features = []
        
        for feat_type in feature_types:
            if feat_type == FeatureType.PIXEL:
                feat = self.extract_pixel_features(image, region)
            elif feat_type == FeatureType.LBP:
                feat = self.extract_lbp_features(image, region)
            elif feat_type == FeatureType.GLCM:
                feat = self.extract_glcm_features(image, region)
            elif feat_type == FeatureType.INTENSITY_STATS:
                feat = self.extract_intensity_stats(image, region)
            elif feat_type == FeatureType.GABOR:
                feat = self.extract_gabor_features(image, region)
            elif feat_type == FeatureType.HOG:
                feat = self.extract_hog_features(image, region)
            else:
                raise ValueError(f"Tipo de feature não suportado: {feat_type}")
            
            all_features.append(feat)
        
        return np.concatenate(all_features)
    
    def extract_from_regions(self, image: np.ndarray,
                              regions: List[Tuple[int, int]],
                              feature_types: Optional[List[FeatureType]] = None
                              ) -> np.ndarray:
        """
        Extrai features de múltiplas regiões e concatena.
        
        Args:
            image: Imagem em escala de cinza
            regions: Lista de regiões (start_col, end_col)
            feature_types: Tipos de features a extrair
            
        Returns:
            Vetor concatenado de features de todas as regiões
        """
        all_features = []
        
        for region in regions:
            feat = self.extract_features(image, region, feature_types)
            all_features.append(feat)
        
        return np.concatenate(all_features)


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_feature_extractor(config: Optional[FeatureConfig] = None) -> FeatureExtractor:
    """Factory function para criar extrator de features."""
    return FeatureExtractor(config)


def extract_all_features(image: np.ndarray, 
                          config: Optional[FeatureConfig] = None
                          ) -> Dict[str, np.ndarray]:
    """
    Extrai todos os tipos de features separadamente.
    
    Útil para análise comparativa de contribuição de cada tipo.
    
    Args:
        image: Imagem em escala de cinza
        config: Configuração de extração
        
    Returns:
        Dicionário com features por tipo
    """
    extractor = FeatureExtractor(config)
    
    results = {}
    for feat_type in FeatureType:
        try:
            results[feat_type.value] = extractor.extract_features(
                image, feature_types=[feat_type]
            )
        except Exception as e:
            results[feat_type.value] = None
            print(f"Erro ao extrair {feat_type.value}: {e}")
    
    return results


def get_feature_dimensions(config: Optional[FeatureConfig] = None,
                            image_shape: Tuple[int, int] = (191, 720)
                            ) -> Dict[str, int]:
    """
    Calcula dimensões esperadas de cada tipo de feature.
    
    Útil para pré-alocação de arrays e verificação de consistência.
    
    Args:
        config: Configuração de extração
        image_shape: Shape da imagem normalizada (height, width)
        
    Returns:
        Dicionário com dimensões por tipo
    """
    config = config or FeatureConfig()
    h, w = image_shape
    
    # Cria imagem dummy para calcular dimensões
    dummy_image = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    extractor = FeatureExtractor(config)
    
    dimensions = {}
    for feat_type in FeatureType:
        try:
            feat = extractor.extract_features(dummy_image, feature_types=[feat_type])
            dimensions[feat_type.value] = len(feat)
        except Exception:
            dimensions[feat_type.value] = 0
    
    return dimensions


# =============================================================================
# Funções de compatibilidade com código antigo
# =============================================================================

def hogFeature(normalizedIrisPatch: np.ndarray, 
               regions: List[Tuple[int, int]]) -> List:
    """Compatibilidade com função antiga."""
    extractor = FeatureExtractor()
    features = []
    upper_cut = 10
    
    for reg in regions:
        cropped = normalizedIrisPatch[upper_cut:, reg[0]:reg[1]]
        feat = extractor.extract_hog_features(cropped)
        features.extend([[f] for f in feat])
    
    return features


def lbpFeature(normalizedIrisPatch: np.ndarray,
               regions: List[Tuple[int, int]]) -> List:
    """Compatibilidade com função antiga."""
    extractor = FeatureExtractor()
    features = []
    upper_cut = 10
    
    for reg in regions:
        cropped = normalizedIrisPatch[upper_cut:, reg[0]:reg[1]]
        feat = extractor.extract_lbp_features(cropped)
        features.extend([[f] for f in feat])
    
    return features


def gaborFeature(normalizedIrisPatch: np.ndarray,
                 regions: List[Tuple[int, int]]) -> List:
    """Compatibilidade com função antiga."""
    extractor = FeatureExtractor()
    features = []
    upper_cut = 10
    
    for reg in regions:
        cropped = normalizedIrisPatch[upper_cut:, reg[0]:reg[1]]
        feat = extractor.extract_gabor_features(cropped)
        features.extend([[f] for f in feat])
    
    return features


def extract_image_feature(image: np.ndarray,
                          regions: List[Tuple[int, int]],
                          downSampleSize: float) -> List:
    """Compatibilidade com função antiga."""
    config = FeatureConfig()
    config.pixel_downsample_size = int(downSampleSize)
    extractor = FeatureExtractor(config)
    
    features = []
    upper_cut = 10
    
    for reg in regions:
        cropped = image[upper_cut:, reg[0]:reg[1]]
        feat = extractor.extract_pixel_features(cropped)
        features.extend([[f] for f in feat])
    
    return features
