"""
Módulo de Métricas de Avaliação
===============================
Implementa métricas de avaliação clínica e estatística.

Métricas conforme metodologia do artigo:
- Accuracy
- Sensibilidade (recall da classe DM2)
- Especificidade
- Precisão
- F1-score
- Matriz de confusão

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from scipy import stats

from .config import MetricsConfig


@dataclass
class MetricsResult:
    """
    Resultado completo de métricas de classificação.
    
    Inclui métricas individuais, matriz de confusão e
    intervalos de confiança quando aplicável.
    """
    # Métricas principais
    accuracy: float = 0.0
    sensitivity: float = 0.0  # recall/TPR
    specificity: float = 0.0  # TNR
    precision: float = 0.0    # PPV
    f1: float = 0.0
    
    # Desvios padrão (para validação cruzada)
    accuracy_std: float = 0.0
    sensitivity_std: float = 0.0
    specificity_std: float = 0.0
    precision_std: float = 0.0
    f1_std: float = 0.0
    
    # Matriz de confusão
    confusion_matrix: Optional[np.ndarray] = None
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    # Métricas adicionais
    npv: float = 0.0  # Negative Predictive Value
    auc: float = 0.0  # Area Under ROC Curve
    
    # Intervalos de confiança (95%)
    ci_lower: Dict[str, float] = field(default_factory=dict)
    ci_upper: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Converte resultado para dicionário."""
        return {
            'accuracy': self.accuracy,
            'accuracy_std': self.accuracy_std,
            'sensitivity': self.sensitivity,
            'sensitivity_std': self.sensitivity_std,
            'specificity': self.specificity,
            'specificity_std': self.specificity_std,
            'precision': self.precision,
            'precision_std': self.precision_std,
            'f1': self.f1,
            'f1_std': self.f1_std,
            'npv': self.npv,
            'auc': self.auc,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'tp': self.tp
        }
    
    def __str__(self) -> str:
        """Representação em string formatada."""
        lines = [
            "=" * 50,
            "MÉTRICAS DE CLASSIFICAÇÃO",
            "=" * 50,
            f"Accuracy:     {self.accuracy:.4f} ± {self.accuracy_std:.4f}",
            f"Sensitivity:  {self.sensitivity:.4f} ± {self.sensitivity_std:.4f}",
            f"Specificity:  {self.specificity:.4f} ± {self.specificity_std:.4f}",
            f"Precision:    {self.precision:.4f} ± {self.precision_std:.4f}",
            f"F1-score:     {self.f1:.4f} ± {self.f1_std:.4f}",
            "-" * 50,
            "Matriz de Confusão:",
            f"  TN: {self.tn}  FP: {self.fp}",
            f"  FN: {self.fn}  TP: {self.tp}",
            "=" * 50
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculadora de métricas de classificação.
    
    Implementa cálculo de métricas clinicamente informativas
    conforme descrito na metodologia do artigo.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Inicializa a calculadora.
        
        Args:
            config: Configuração de métricas
        """
        self.config = config or MetricsConfig()
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None) -> MetricsResult:
        """
        Calcula todas as métricas para predições.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            y_proba: Probabilidades (opcional, para AUC)
            
        Returns:
            MetricsResult com todas as métricas
        """
        result = MetricsResult()
        
        # Métricas básicas
        result.accuracy = accuracy_score(y_true, y_pred)
        result.precision = precision_score(y_true, y_pred, zero_division=0)
        result.sensitivity = recall_score(y_true, y_pred, zero_division=0)
        result.f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        result.confusion_matrix = cm
        
        if cm.shape == (2, 2):
            result.tn, result.fp, result.fn, result.tp = cm.ravel()
            
            # Especificidade: TN / (TN + FP)
            if (result.tn + result.fp) > 0:
                result.specificity = result.tn / (result.tn + result.fp)
            
            # NPV: TN / (TN + FN)
            if (result.tn + result.fn) > 0:
                result.npv = result.tn / (result.tn + result.fn)
        
        # AUC (se probabilidades disponíveis)
        if y_proba is not None:
            try:
                result.auc = roc_auc_score(y_true, y_proba)
            except Exception:
                result.auc = 0.0
        
        return result
    
    def calculate_from_folds(self, 
                              accuracy_folds: List[float],
                              sensitivity_folds: List[float],
                              specificity_folds: List[float],
                              precision_folds: List[float],
                              f1_folds: List[float],
                              y_true: Optional[np.ndarray] = None,
                              y_pred: Optional[np.ndarray] = None
                              ) -> MetricsResult:
        """
        Calcula métricas agregadas a partir de resultados por fold.
        
        Args:
            accuracy_folds: Acurácia por fold
            sensitivity_folds: Sensibilidade por fold
            specificity_folds: Especificidade por fold
            precision_folds: Precisão por fold
            f1_folds: F1 por fold
            y_true: Labels verdadeiros (para matriz de confusão)
            y_pred: Predições (para matriz de confusão)
            
        Returns:
            MetricsResult agregado
        """
        result = MetricsResult()
        
        # Médias
        result.accuracy = np.mean(accuracy_folds)
        result.sensitivity = np.mean(sensitivity_folds)
        result.specificity = np.mean(specificity_folds)
        result.precision = np.mean(precision_folds)
        result.f1 = np.mean(f1_folds)
        
        # Desvios padrão
        result.accuracy_std = np.std(accuracy_folds)
        result.sensitivity_std = np.std(sensitivity_folds)
        result.specificity_std = np.std(specificity_folds)
        result.precision_std = np.std(precision_folds)
        result.f1_std = np.std(f1_folds)
        
        # Matriz de confusão agregada
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            result.confusion_matrix = cm
            
            if cm.shape == (2, 2):
                result.tn, result.fp, result.fn, result.tp = cm.ravel()
        
        # Intervalos de confiança
        if self.config.compute_confidence_interval:
            result.ci_lower, result.ci_upper = self._compute_confidence_intervals(
                accuracy_folds, sensitivity_folds, specificity_folds,
                precision_folds, f1_folds
            )
        
        return result
    
    def _compute_confidence_intervals(self,
                                       accuracy_folds: List[float],
                                       sensitivity_folds: List[float],
                                       specificity_folds: List[float],
                                       precision_folds: List[float],
                                       f1_folds: List[float]
                                       ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calcula intervalos de confiança para métricas.
        
        Usa distribuição t de Student para amostras pequenas (K folds).
        
        Returns:
            Tupla (ci_lower, ci_upper) como dicionários
        """
        ci_lower = {}
        ci_upper = {}
        
        metrics = {
            'accuracy': accuracy_folds,
            'sensitivity': sensitivity_folds,
            'specificity': specificity_folds,
            'precision': precision_folds,
            'f1': f1_folds
        }
        
        confidence = self.config.confidence_level
        
        for name, values in metrics.items():
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                n = len(values)
                
                # t-score para intervalo de confiança
                t_score = stats.t.ppf((1 + confidence) / 2, n - 1)
                margin = t_score * std / np.sqrt(n)
                
                ci_lower[name] = max(0, mean - margin)
                ci_upper[name] = min(1, mean + margin)
            else:
                ci_lower[name] = values[0] if values else 0
                ci_upper[name] = values[0] if values else 0
        
        return ci_lower, ci_upper


def compute_confusion_matrix(y_true: np.ndarray, 
                              y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calcula matriz de confusão e retorna como dicionário.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        
    Returns:
        Dicionário com TN, FP, FN, TP
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    else:
        return {'matrix': cm.tolist()}


def format_metrics_table(results: Dict[str, MetricsResult],
                          include_std: bool = True) -> str:
    """
    Formata resultados como tabela texto.
    
    Args:
        results: Dicionário de resultados por classificador
        include_std: Se True, inclui desvio padrão
        
    Returns:
        String formatada como tabela
    """
    lines = []
    
    # Header
    if include_std:
        header = f"{'Classifier':<15} {'Accuracy':<18} {'Sensitivity':<18} {'Specificity':<18} {'F1-score':<18}"
    else:
        header = f"{'Classifier':<15} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-score':<10}"
    
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))
    
    # Rows
    for name, result in results.items():
        if include_std:
            line = (f"{name:<15} "
                   f"{result.accuracy:.4f}±{result.accuracy_std:.3f}  "
                   f"{result.sensitivity:.4f}±{result.sensitivity_std:.3f}  "
                   f"{result.specificity:.4f}±{result.specificity_std:.3f}  "
                   f"{result.f1:.4f}±{result.f1_std:.3f}")
        else:
            line = (f"{name:<15} "
                   f"{result.accuracy:.4f}     "
                   f"{result.sensitivity:.4f}       "
                   f"{result.specificity:.4f}       "
                   f"{result.f1:.4f}")
        lines.append(line)
    
    lines.append("=" * len(header))
    
    return "\n".join(lines)


def format_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             labels: List[str] = None) -> str:
    """
    Formata matriz de confusão como string.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        labels: Nomes das classes
        
    Returns:
        String formatada
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or ['Control', 'DM2']
    
    lines = [
        "\nMatriz de Confusão:",
        "=" * 40,
        f"{'':>15} {'Predito':<20}",
        f"{'':>15} {labels[0]:<10} {labels[1]:<10}",
        "-" * 40,
    ]
    
    if cm.shape == (2, 2):
        lines.append(f"{'Real ' + labels[0]:<15} {cm[0,0]:<10} {cm[0,1]:<10}")
        lines.append(f"{'Real ' + labels[1]:<15} {cm[1,0]:<10} {cm[1,1]:<10}")
    
    lines.append("=" * 40)
    
    # Interpretação
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        lines.extend([
            "\nInterpretação:",
            f"  - Verdadeiros Negativos (TN): {tn}",
            f"  - Falsos Positivos (FP): {fp}",
            f"  - Falsos Negativos (FN): {fn}",
            f"  - Verdadeiros Positivos (TP): {tp}",
        ])
    
    return "\n".join(lines)


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_calculator(config: Optional[MetricsConfig] = None) -> MetricsCalculator:
    """Factory function para criar calculadora."""
    return MetricsCalculator(config)


def quick_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas básicas rapidamente.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        
    Returns:
        Dicionário com métricas básicas
    """
    calculator = MetricsCalculator()
    result = calculator.calculate(y_true, y_pred)
    
    return {
        'accuracy': result.accuracy,
        'sensitivity': result.sensitivity,
        'specificity': result.specificity,
        'precision': result.precision,
        'f1': result.f1
    }


def get_best_classifier(results: Dict[str, MetricsResult],
                         metric: str = 'accuracy') -> Tuple[str, float]:
    """
    Identifica o melhor classificador por uma métrica.
    
    Args:
        results: Dicionário de resultados
        metric: Métrica para comparação
        
    Returns:
        Tupla (nome_classificador, valor_métrica)
    """
    best_name = None
    best_value = -1
    
    for name, result in results.items():
        value = getattr(result, metric, 0)
        if value > best_value:
            best_value = value
            best_name = name
    
    return best_name, best_value


def get_classification_report(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               target_names: List[str] = None) -> str:
    """
    Gera relatório de classificação completo.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        target_names: Nomes das classes
        
    Returns:
        Relatório formatado
    """
    target_names = target_names or ['Control', 'DM2']
    return classification_report(y_true, y_pred, target_names=target_names)
