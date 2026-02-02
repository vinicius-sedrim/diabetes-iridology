"""
Módulo de Geração de Resultados e Visualizações
===============================================
Gera tabelas, gráficos e relatórios conforme metodologia do artigo.

Formatos de saída:
- Tabelas LaTeX (para inclusão no artigo)
- Gráficos matplotlib (heatmaps, barras, curvas)
- Relatórios JSON para análise posterior

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Tenta importar matplotlib (opcional para headless servers)
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib não disponível. Gráficos desabilitados.")

from .classifiers import ClassificationResult
from .local_analysis import SectorAnalysisResult, GridAnalysisResult


class ResultsGenerator:
    """
    Gerador de resultados e visualizações.
    
    Produz tabelas e gráficos no formato adequado para
    inclusão em artigos científicos.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Inicializa o gerador.
        
        Args:
            output_dir: Diretório para salvar outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Tabelas LaTeX
    # =========================================================================
    
    def generate_accuracy_table_latex(self, 
                                       results: Dict[str, Dict[str, ClassificationResult]],
                                       caption: str = "Acurácia por dataset e classificador",
                                       label: str = "tab:accuracy"
                                       ) -> str:
        """
        Gera Tabela 1 do artigo (acurácia por dataset e classificador).
        
        Args:
            results: Resultados por dataset e classificador
            caption: Legenda da tabela
            label: Label para referência LaTeX
            
        Returns:
            String com código LaTeX
        """
        classifiers = ['AdaBoost', 'MLP', 'RandomForest', 'SVM', 'LogisticRegression']
        datasets = ['all', 'L_split', 'R_split', 'personBase', 'personBase_invert']
        
        # Mapeamento de nomes
        clf_display = {
            'AdaBoost': 'AdaBoost',
            'MLP': 'MLP',
            'RandomForest': 'RF',
            'SVM': 'SVM',
            'LogisticRegression': 'LR'
        }
        
        dataset_display = {
            'all': r'\texttt{all}',
            'L_split': r'\texttt{L\_split}',
            'R_split': r'\texttt{R\_split}',
            'personBase': r'\texttt{personBase}',
            'personBase_invert': r'\texttt{personBase\_invert}'
        }
        
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{l' + 'S' * len(classifiers) + '}',
            r'\toprule',
        ]
        
        # Header
        header = '{Dataset}'
        for clf in classifiers:
            header += f' & {{{clf_display[clf]}}}'
        header += r' \\'
        lines.append(header)
        lines.append(r'\midrule')
        
        # Encontra melhor valor para cada dataset
        best_per_dataset = {}
        for dataset in datasets:
            if dataset in results:
                best_acc = 0
                for clf in classifiers:
                    if clf in results[dataset]:
                        acc = results[dataset][clf].accuracy_mean
                        if acc > best_acc:
                            best_acc = acc
                best_per_dataset[dataset] = best_acc
        
        # Rows
        for dataset in datasets:
            row = dataset_display[dataset]
            
            if dataset in results:
                for clf in classifiers:
                    if clf in results[dataset]:
                        acc = results[dataset][clf].accuracy_mean
                        # Destaca o melhor
                        if acc == best_per_dataset.get(dataset, 0):
                            row += f' & \\textbf{{{acc:.4f}}}'
                        else:
                            row += f' & {acc:.4f}'
                    else:
                        row += ' & {--}'
            else:
                row += ' & {--}' * len(classifiers)
            
            row += r' \\'
            lines.append(row)
        
        lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}'
        ])
        
        return '\n'.join(lines)
    
    def generate_top5_table_latex(self,
                                   results: Dict[str, Dict[str, ClassificationResult]],
                                   caption: str = "Top 5 combinações por acurácia",
                                   label: str = "tab:top5"
                                   ) -> str:
        """
        Gera Tabela 3 do artigo (top 5 combinações).
        
        Args:
            results: Resultados por dataset e classificador
            caption: Legenda
            label: Label LaTeX
            
        Returns:
            Código LaTeX
        """
        # Coleta todos os resultados
        all_results = []
        for dataset, clf_results in results.items():
            for clf_name, result in clf_results.items():
                all_results.append({
                    'dataset': dataset,
                    'classifier': clf_name,
                    'accuracy': result.accuracy_mean,
                    'std': result.accuracy_std
                })
        
        # Ordena por acurácia
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        top5 = all_results[:5]
        
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{cllcc}',
            r'\toprule',
            r'\textbf{Posição} & \textbf{Dataset} & \textbf{Classificador} & \textbf{Acurácia (\%)} & \textbf{Desvio padrão (pp)} \\',
            r'\midrule',
        ]
        
        for i, res in enumerate(top5):
            dataset = res['dataset'].replace('_', r'\_')
            lines.append(
                f"{i+1} & \\texttt{{{dataset}}} & {res['classifier']} & "
                f"{res['accuracy']*100:.2f} & {res['std']*100:.2f} \\\\"
            )
        
        lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}'
        ])
        
        return '\n'.join(lines)
    
    def generate_robustness_table_latex(self,
                                         results: Dict[str, Dict[str, ClassificationResult]],
                                         caption: str = "Comparação entre dados originais e transformados",
                                         label: str = "tab:robustness"
                                         ) -> str:
        """
        Gera Tabela 4 do artigo (robustez fotométrica).
        
        Args:
            results: Resultados por transformação
            caption: Legenda
            label: Label LaTeX
            
        Returns:
            Código LaTeX
        """
        transforms = ['original', 'histogram', 'clahe', 'blur']
        transform_display = {
            'original': 'Original',
            'histogram': 'Histogram',
            'clahe': 'CLAHE',
            'blur': 'Blur'
        }
        
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            r'\small',
            r'\begin{tabular}{l' + 'c' * len(transforms) + '}',
            r'\toprule',
        ]
        
        # Header
        header = r'\textbf{Dataset}'
        for t in transforms:
            header += f' & \\textbf{{{transform_display[t]}}}'
        header += r' \\'
        lines.append(header)
        lines.append(r'\midrule')
        
        # Calcula média por transformação (usando LR como referência)
        for t in transforms:
            if t in results:
                # Média de acurácia de todos os classificadores
                accs = [r.accuracy_mean for r in results[t].values()]
                mean_acc = np.mean(accs) * 100
                lines.append(f"Média & {mean_acc:.2f}\\% \\\\")
        
        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}'
        ])
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Gráficos
    # =========================================================================
    
    def plot_sector_analysis(self, result: SectorAnalysisResult,
                              metric: str = 'accuracy',
                              save_path: Optional[str] = None) -> None:
        """
        Plota resultados de análise por setores.
        
        Args:
            result: Resultado da análise por setores
            metric: Métrica a plotar
            save_path: Caminho para salvar figura
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib não disponível")
            return
        
        angles = np.array(result.sector_start_degrees)
        
        if metric == 'accuracy':
            values = np.array(result.accuracies)
        elif metric == 'sensitivity':
            values = np.array(result.sensitivities)
        elif metric == 'specificity':
            values = np.array(result.specificities)
        else:
            values = np.array(result.f1_scores)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.bar(angles, values, width=8, color='steelblue', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        ax.axhline(y=np.mean(values), color='green', linestyle='-', alpha=0.7, 
                   label=f'Média ({np.mean(values):.3f})')
        
        ax.set_xlabel('Ângulo (graus)')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} por Setor Angular - {result.classifier_name}')
        ax.set_xlim(-5, 365)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pancreatic_heatmap(self, result: GridAnalysisResult,
                                 metric: str = 'accuracy',
                                 save_path: Optional[str] = None) -> None:
        """
        Plota heatmap da análise pancreática.
        
        Args:
            result: Resultado da análise por malha
            metric: Métrica a plotar
            save_path: Caminho para salvar
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib não disponível")
            return
        
        grid = getattr(result, f'{metric}_grid', result.accuracy_grid)
        
        if grid is None:
            print(f"Grid de {metric} não disponível")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(grid, cmap='RdYlGn', vmin=0.4, vmax=0.9, aspect='auto')
        
        # Adiciona colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.capitalize())
        
        # Labels
        ax.set_xlabel('Coluna')
        ax.set_ylabel('Linha')
        ax.set_title(f'Heatmap {metric.capitalize()} - Região Pancreática\n'
                    f'{result.classifier_name} - {result.region_name}')
        
        # Adiciona valores nas células
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text = ax.text(j, i, f'{grid[i,j]:.2f}',
                              ha='center', va='center', fontsize=6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap salvo em: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_classifier_comparison(self, 
                                    results: Dict[str, Dict[str, ClassificationResult]],
                                    save_path: Optional[str] = None) -> None:
        """
        Plota comparação de classificadores por dataset.
        
        Args:
            results: Resultados por dataset e classificador
            save_path: Caminho para salvar
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib não disponível")
            return
        
        datasets = list(results.keys())
        classifiers = ['AdaBoost', 'MLP', 'RandomForest', 'SVM', 'LogisticRegression']
        
        # Prepara dados
        data = np.zeros((len(datasets), len(classifiers)))
        for i, dataset in enumerate(datasets):
            for j, clf in enumerate(classifiers):
                if clf in results.get(dataset, {}):
                    data[i, j] = results[dataset][clf].accuracy_mean
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.15
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(classifiers)))
        
        for i, clf in enumerate(classifiers):
            offset = (i - len(classifiers)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[:, i], width, label=clf, color=colors[i])
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Acurácia')
        ax.set_title('Comparação de Classificadores por Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparação salva em: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # =========================================================================
    # Exportação
    # =========================================================================
    
    def export_results_json(self, results: dict, filename: str) -> str:
        """
        Exporta resultados para JSON.
        
        Args:
            results: Dicionário de resultados
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Converte resultados para formato serializável
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        serializable = convert(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados exportados para: {filepath}")
        return filepath
    
    def export_latex_tables(self, 
                             results: Dict[str, Dict[str, ClassificationResult]],
                             robustness_results: Optional[Dict] = None,
                             filename: str = "tables.tex") -> str:
        """
        Exporta todas as tabelas LaTeX.
        
        Args:
            results: Resultados principais
            robustness_results: Resultados de robustez (opcional)
            filename: Nome do arquivo
            
        Returns:
            Caminho do arquivo salvo
        """
        filepath = os.path.join(self.output_dir, filename)
        
        content = [
            "% Tabelas geradas automaticamente",
            f"% Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "% Tabela 1: Acurácia por dataset e classificador",
            self.generate_accuracy_table_latex(results),
            "",
            "% Tabela 2: Top 5 combinações",
            self.generate_top5_table_latex(results),
        ]
        
        if robustness_results:
            content.extend([
                "",
                "% Tabela 3: Robustez fotométrica",
                self.generate_robustness_table_latex(robustness_results),
            ])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"Tabelas LaTeX exportadas para: {filepath}")
        return filepath
    
    def generate_full_report(self, 
                              results: Dict,
                              experiment_name: str = "iridology_analysis"
                              ) -> str:
        """
        Gera relatório completo em formato texto.
        
        Args:
            results: Todos os resultados
            experiment_name: Nome do experimento
            
        Returns:
            Caminho do arquivo salvo
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        
        lines = [
            "=" * 70,
            f"RELATÓRIO: {experiment_name}",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]
        
        # Adiciona seções baseado nos resultados disponíveis
        if 'all_datasets' in results:
            lines.append("\n" + "=" * 50)
            lines.append("RESULTADOS POR DATASET")
            lines.append("=" * 50)
            
            for dataset_name, clf_results in results['all_datasets'].items():
                lines.append(f"\n--- {dataset_name} ---")
                for clf_name, result in clf_results.items():
                    lines.append(
                        f"  {clf_name}: "
                        f"Acc={result.accuracy_mean:.4f}±{result.accuracy_std:.4f}, "
                        f"Sens={result.sensitivity_mean:.4f}, "
                        f"Spec={result.specificity_mean:.4f}, "
                        f"F1={result.f1_mean:.4f}"
                    )
        
        if 'robustness' in results:
            lines.append("\n" + "=" * 50)
            lines.append("TESTE DE ROBUSTEZ FOTOMÉTRICA")
            lines.append("=" * 50)
            
            for transform_name, clf_results in results['robustness'].items():
                lines.append(f"\n--- {transform_name} ---")
                for clf_name, result in clf_results.items():
                    lines.append(f"  {clf_name}: Acc={result.accuracy_mean:.4f}")
        
        lines.append("\n" + "=" * 70)
        lines.append("FIM DO RELATÓRIO")
        lines.append("=" * 70)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Relatório salvo em: {filepath}")
        return filepath


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_results_generator(output_dir: str = "results") -> ResultsGenerator:
    """Factory function para criar gerador de resultados."""
    return ResultsGenerator(output_dir)


def quick_summary(results: Dict[str, Dict[str, ClassificationResult]]) -> str:
    """
    Gera resumo rápido dos resultados.
    
    Args:
        results: Resultados por dataset e classificador
        
    Returns:
        String com resumo
    """
    lines = ["Resumo de Resultados:", "-" * 40]
    
    best_overall = {'acc': 0, 'dataset': '', 'clf': ''}
    
    for dataset, clf_results in results.items():
        for clf_name, result in clf_results.items():
            if result.accuracy_mean > best_overall['acc']:
                best_overall = {
                    'acc': result.accuracy_mean,
                    'dataset': dataset,
                    'clf': clf_name
                }
    
    lines.append(f"Melhor resultado: {best_overall['clf']} em {best_overall['dataset']}")
    lines.append(f"Acurácia: {best_overall['acc']:.4f}")
    
    return '\n'.join(lines)
