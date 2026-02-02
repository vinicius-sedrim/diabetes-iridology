"""
Módulo de Análise Local
=======================
Implementa análises locais por região da íris.

Análises conforme metodologia do artigo:
- Análise por setores angulares (passos de 10°)
- Análise de região pancreática (malha 12x12)
- Geração de heatmaps de desempenho

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from .config import (
    LocalAnalysisConfig, 
    ClassifierConfig, 
    FeatureConfig,
    ClassifierType,
    FeatureType
)
from .classifiers import CrossValidator, ClassifierFactory
from .feature_extraction import FeatureExtractor
from .metrics import MetricsCalculator, MetricsResult


@dataclass
class SectorAnalysisResult:
    """
    Resultado de análise por setor angular.
    
    Armazena métricas por setor para identificar regiões
    com maior poder discriminativo.
    """
    sector_start_degrees: List[int] = field(default_factory=list)
    sector_end_degrees: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    sensitivities: List[float] = field(default_factory=list)
    specificities: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    
    # Informações adicionais
    classifier_name: str = ""
    dataset_name: str = ""
    
    def get_best_sector(self, metric: str = 'accuracy') -> Tuple[int, int, float]:
        """
        Retorna o setor com melhor desempenho.
        
        Args:
            metric: Métrica para comparação
            
        Returns:
            Tupla (start_deg, end_deg, value)
        """
        values = getattr(self, f'{metric}' if metric == 'accuracies' else f'{metric}s', self.accuracies)
        best_idx = np.argmax(values)
        
        return (
            self.sector_start_degrees[best_idx],
            self.sector_end_degrees[best_idx],
            values[best_idx]
        )
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'sectors': list(zip(self.sector_start_degrees, self.sector_end_degrees)),
            'accuracies': self.accuracies,
            'sensitivities': self.sensitivities,
            'specificities': self.specificities,
            'f1_scores': self.f1_scores,
            'classifier': self.classifier_name,
            'dataset': self.dataset_name
        }


@dataclass
class GridAnalysisResult:
    """
    Resultado de análise por malha (grid).
    
    Usado para análise de região pancreática com malha 12x12.
    """
    grid_shape: Tuple[int, int] = (12, 12)
    accuracy_grid: Optional[np.ndarray] = None
    sensitivity_grid: Optional[np.ndarray] = None
    specificity_grid: Optional[np.ndarray] = None
    f1_grid: Optional[np.ndarray] = None
    
    # Informações adicionais
    classifier_name: str = ""
    dataset_name: str = ""
    region_name: str = ""
    
    def get_best_cell(self, metric: str = 'accuracy') -> Tuple[int, int, float]:
        """
        Retorna a célula com melhor desempenho.
        
        Args:
            metric: Métrica para comparação
            
        Returns:
            Tupla (row, col, value)
        """
        grid = getattr(self, f'{metric}_grid', self.accuracy_grid)
        if grid is None:
            return (0, 0, 0.0)
        
        best_idx = np.unravel_index(np.argmax(grid), grid.shape)
        return (best_idx[0], best_idx[1], grid[best_idx])


class SectorAnalyzer:
    """
    Analisador de setores angulares da íris.
    
    Varre a íris normalizada em setores de tamanho configurável
    (padrão: 10°) para identificar regiões com padrões discriminativos.
    """
    
    def __init__(self, 
                 local_config: Optional[LocalAnalysisConfig] = None,
                 classifier_config: Optional[ClassifierConfig] = None,
                 feature_config: Optional[FeatureConfig] = None):
        """
        Inicializa o analisador.
        
        Args:
            local_config: Configuração de análise local
            classifier_config: Configuração de classificadores
            feature_config: Configuração de features
        """
        self.local_config = local_config or LocalAnalysisConfig()
        self.classifier_config = classifier_config or ClassifierConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        self.validator = CrossValidator(self.classifier_config)
        self.extractor = FeatureExtractor(self.feature_config)
    
    def _degrees_to_pixels(self, degrees: int, iris_width: int = 720) -> int:
        """
        Converte graus para pixels na íris normalizada.
        
        Args:
            degrees: Ângulo em graus
            iris_width: Largura da íris normalizada
            
        Returns:
            Posição em pixels
        """
        return int(degrees * iris_width / 360)
    
    def _get_sector_regions(self, iris_width: int = 720) -> List[Tuple[int, int, int, int]]:
        """
        Gera lista de regiões setoriais.
        
        Returns:
            Lista de tuplas (start_deg, end_deg, start_px, end_px)
        """
        step = self.local_config.sector_step_degrees
        width = self.local_config.sector_width_degrees
        
        regions = []
        for start_deg in range(0, 360 - width + 1, step):
            end_deg = start_deg + width
            start_px = self._degrees_to_pixels(start_deg, iris_width)
            end_px = self._degrees_to_pixels(end_deg, iris_width)
            regions.append((start_deg, end_deg, start_px, end_px))
        
        return regions
    
    def analyze(self, images: np.ndarray, labels: np.ndarray,
                classifier_type: ClassifierType = ClassifierType.LR,
                dataset_name: str = "unknown",
                feature_types: Optional[List[FeatureType]] = None
                ) -> SectorAnalysisResult:
        """
        Analisa desempenho por setor angular.
        
        Args:
            images: Array de imagens normalizadas (N, H, W)
            labels: Array de labels (N,)
            classifier_type: Tipo de classificador
            dataset_name: Nome do dataset
            feature_types: Tipos de features a usar
            
        Returns:
            SectorAnalysisResult com métricas por setor
        """
        feature_types = feature_types or [FeatureType.PIXEL]
        iris_width = images.shape[2] if len(images.shape) == 3 else images.shape[1]
        
        result = SectorAnalysisResult(
            classifier_name=classifier_type.value,
            dataset_name=dataset_name
        )
        
        regions = self._get_sector_regions(iris_width)
        
        print(f"Analisando {len(regions)} setores...")
        
        for start_deg, end_deg, start_px, end_px in regions:
            # Extrai features do setor
            features = []
            for img in images:
                if len(img.shape) == 3:
                    img = img[:, :, 0]  # Usa primeiro canal se colorida
                
                feat = self.extractor.extract_features(
                    img, 
                    region=(start_px, end_px),
                    feature_types=feature_types
                )
                features.append(feat)
            
            X = np.array(features, dtype=np.float32)
            y = np.array(labels, dtype=np.int32)
            
            # Valida
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_result = self.validator.validate(X, y, classifier_type, dataset_name)
            
            # Armazena resultados
            result.sector_start_degrees.append(start_deg)
            result.sector_end_degrees.append(end_deg)
            result.accuracies.append(cv_result.accuracy_mean)
            result.sensitivities.append(cv_result.sensitivity_mean)
            result.specificities.append(cv_result.specificity_mean)
            result.f1_scores.append(cv_result.f1_mean)
            
            print(f"  Setor {start_deg}°-{end_deg}°: Acc={cv_result.accuracy_mean:.4f}")
        
        return result


class PancreaticRegionAnalyzer:
    """
    Analisador de região pancreática com malha 12x12.
    
    Subdivide a região associada ao pâncreas em uma grade
    para análise espacial detalhada.
    """
    
    def __init__(self,
                 local_config: Optional[LocalAnalysisConfig] = None,
                 classifier_config: Optional[ClassifierConfig] = None,
                 feature_config: Optional[FeatureConfig] = None):
        """
        Inicializa o analisador.
        
        Args:
            local_config: Configuração de análise local
            classifier_config: Configuração de classificadores
            feature_config: Configuração de features
        """
        self.local_config = local_config or LocalAnalysisConfig()
        self.classifier_config = classifier_config or ClassifierConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        self.validator = CrossValidator(self.classifier_config)
        self.extractor = FeatureExtractor(self.feature_config)
        
        self.grid_shape = self.local_config.pancreatic_grid_size
    
    def _get_pancreatic_region(self, eye_side: str = 'left',
                                iris_width: int = 720,
                                iris_height: int = 191) -> Tuple[int, int, int, int]:
        """
        Retorna limites da região pancreática.
        
        Args:
            eye_side: 'left' ou 'right'
            iris_width: Largura da íris normalizada
            iris_height: Altura da íris normalizada
            
        Returns:
            Tupla (row_start, row_end, col_start, col_end)
        """
        # Região pancreática conforme mapas iridológicos
        # Olho esquerdo: ~215-235° (aproximadamente 430-470 pixels para iris de 720px)
        # Olho direito: ~125-145° (aproximadamente 250-290 pixels)
        
        if eye_side == 'left':
            regions = self.local_config.predefined_regions.get('pancreas_left', [(430, 470)])
        else:
            regions = self.local_config.predefined_regions.get('pancreas_right', [(250, 290)])
        
        col_start, col_end = regions[0]
        
        # Usa toda a altura da íris (exceto região superior cortada)
        row_start = 10  # upper_cut_height
        row_end = iris_height
        
        return (row_start, row_end, col_start, col_end)
    
    def analyze(self, images: np.ndarray, labels: np.ndarray,
                classifier_type: ClassifierType = ClassifierType.LR,
                eye_side: str = 'left',
                dataset_name: str = "unknown",
                feature_types: Optional[List[FeatureType]] = None
                ) -> GridAnalysisResult:
        """
        Analisa região pancreática com malha 12x12.
        
        Args:
            images: Array de imagens normalizadas (N, H, W)
            labels: Array de labels (N,)
            classifier_type: Tipo de classificador
            eye_side: 'left' ou 'right'
            dataset_name: Nome do dataset
            feature_types: Tipos de features
            
        Returns:
            GridAnalysisResult com métricas por célula
        """
        feature_types = feature_types or [FeatureType.PIXEL]
        
        iris_height = images.shape[1] if len(images.shape) == 3 else images.shape[0]
        iris_width = images.shape[2] if len(images.shape) == 3 else images.shape[1]
        
        # Define região pancreática
        row_start, row_end, col_start, col_end = self._get_pancreatic_region(
            eye_side, iris_width, iris_height
        )
        
        result = GridAnalysisResult(
            grid_shape=self.grid_shape,
            classifier_name=classifier_type.value,
            dataset_name=dataset_name,
            region_name=f"pancreas_{eye_side}"
        )
        
        # Inicializa grids
        n_rows, n_cols = self.grid_shape
        result.accuracy_grid = np.zeros((n_rows, n_cols))
        result.sensitivity_grid = np.zeros((n_rows, n_cols))
        result.specificity_grid = np.zeros((n_rows, n_cols))
        result.f1_grid = np.zeros((n_rows, n_cols))
        
        # Calcula tamanho de cada célula
        region_height = row_end - row_start
        region_width = col_end - col_start
        cell_height = region_height // n_rows
        cell_width = region_width // n_cols
        
        print(f"Analisando região pancreática ({eye_side}) com malha {n_rows}x{n_cols}...")
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Define limites da célula
                cell_row_start = row_start + i * cell_height
                cell_row_end = row_start + (i + 1) * cell_height
                cell_col_start = col_start + j * cell_width
                cell_col_end = col_start + (j + 1) * cell_width
                
                # Extrai features da célula
                features = []
                for img in images:
                    if len(img.shape) == 3:
                        img = img[:, :, 0]
                    
                    # Extrai região da célula
                    cell_img = img[cell_row_start:cell_row_end, cell_col_start:cell_col_end]
                    
                    # Extrai features (usa estatísticas para células pequenas)
                    feat = self.extractor.extract_intensity_stats(cell_img)
                    features.append(feat)
                
                X = np.array(features, dtype=np.float32)
                y = np.array(labels, dtype=np.int32)
                
                # Valida
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        cv_result = self.validator.validate(X, y, classifier_type, dataset_name)
                        
                        result.accuracy_grid[i, j] = cv_result.accuracy_mean
                        result.sensitivity_grid[i, j] = cv_result.sensitivity_mean
                        result.specificity_grid[i, j] = cv_result.specificity_mean
                        result.f1_grid[i, j] = cv_result.f1_mean
                    except Exception as e:
                        print(f"  Erro na célula ({i},{j}): {e}")
                        result.accuracy_grid[i, j] = 0.5
        
        print(f"  Melhor célula: {result.get_best_cell()}")
        
        return result


class LocalAnalysisPipeline:
    """
    Pipeline completo de análise local.
    
    Combina análise por setores e análise de região pancreática
    em um único fluxo de execução.
    """
    
    def __init__(self,
                 local_config: Optional[LocalAnalysisConfig] = None,
                 classifier_config: Optional[ClassifierConfig] = None,
                 feature_config: Optional[FeatureConfig] = None):
        """
        Inicializa o pipeline.
        
        Args:
            local_config: Configuração de análise local
            classifier_config: Configuração de classificadores
            feature_config: Configuração de features
        """
        self.sector_analyzer = SectorAnalyzer(
            local_config, classifier_config, feature_config
        )
        self.pancreatic_analyzer = PancreaticRegionAnalyzer(
            local_config, classifier_config, feature_config
        )
    
    def run_full_analysis(self, images: np.ndarray, labels: np.ndarray,
                          classifier_type: ClassifierType = ClassifierType.LR,
                          eye_side: str = 'left',
                          dataset_name: str = "unknown"
                          ) -> Dict[str, any]:
        """
        Executa análise local completa.
        
        Args:
            images: Array de imagens normalizadas
            labels: Array de labels
            classifier_type: Tipo de classificador
            eye_side: Lado do olho
            dataset_name: Nome do dataset
            
        Returns:
            Dicionário com resultados de todas as análises
        """
        results = {}
        
        print("\n" + "=" * 50)
        print("ANÁLISE POR SETORES ANGULARES")
        print("=" * 50)
        results['sector_analysis'] = self.sector_analyzer.analyze(
            images, labels, classifier_type, dataset_name
        )
        
        print("\n" + "=" * 50)
        print("ANÁLISE DE REGIÃO PANCREÁTICA")
        print("=" * 50)
        results['pancreatic_analysis'] = self.pancreatic_analyzer.analyze(
            images, labels, classifier_type, eye_side, dataset_name
        )
        
        return results


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_sector_analyzer(config: Optional[LocalAnalysisConfig] = None) -> SectorAnalyzer:
    """Factory function para criar analisador de setores."""
    return SectorAnalyzer(config)


def create_pancreatic_analyzer(config: Optional[LocalAnalysisConfig] = None) -> PancreaticRegionAnalyzer:
    """Factory function para criar analisador pancreático."""
    return PancreaticRegionAnalyzer(config)


def generate_sector_heatmap_data(result: SectorAnalysisResult) -> Dict[str, np.ndarray]:
    """
    Gera dados para heatmap de setores.
    
    Args:
        result: Resultado de análise por setores
        
    Returns:
        Dicionário com arrays para plotting
    """
    return {
        'angles': np.array(result.sector_start_degrees),
        'accuracy': np.array(result.accuracies),
        'sensitivity': np.array(result.sensitivities),
        'specificity': np.array(result.specificities),
        'f1': np.array(result.f1_scores)
    }


def generate_grid_heatmap_data(result: GridAnalysisResult) -> Dict[str, np.ndarray]:
    """
    Gera dados para heatmap de malha.
    
    Args:
        result: Resultado de análise por malha
        
    Returns:
        Dicionário com grids para plotting
    """
    return {
        'accuracy': result.accuracy_grid,
        'sensitivity': result.sensitivity_grid,
        'specificity': result.specificity_grid,
        'f1': result.f1_grid
    }


def find_discriminative_regions(sector_result: SectorAnalysisResult,
                                 threshold: float = 0.7) -> List[Tuple[int, int, float]]:
    """
    Identifica regiões com desempenho acima do limiar.
    
    Args:
        sector_result: Resultado de análise por setores
        threshold: Limiar de acurácia
        
    Returns:
        Lista de regiões discriminativas (start, end, accuracy)
    """
    regions = []
    
    for i, acc in enumerate(sector_result.accuracies):
        if acc >= threshold:
            regions.append((
                sector_result.sector_start_degrees[i],
                sector_result.sector_end_degrees[i],
                acc
            ))
    
    return sorted(regions, key=lambda x: x[2], reverse=True)
