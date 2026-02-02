"""
Pipeline Principal de Análise de Iridologia
============================================
Implementa o pipeline completo e auditável de análise.

Funcionalidades conforme metodologia do artigo:
- Segmentação de íris e pupila (active contours)
- Normalização geométrica (rubber sheet model)
- Carregamento de dados por dataset
- Extração de features
- Treinamento e validação cruzada estratificada
- Análises globais e locais
- Testes de robustez com transformações fotométricas
- Geração de resultados tabulares

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"

Uso:
    python main.py --dataset personBase --channel gray
    python main.py --run-all
    python main.py --robustness-test
    python main.py --process-raw --raw-path /path/to/images
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Importa módulos do projeto
from src.config import (
    ExperimentConfig, 
    DatasetType, 
    ClassifierType, 
    PhotometricTransform,
    FeatureType,
    ColorChannel,
    get_default_config
)
from src.segmentation import IrisSegmenter, SegmentationResult, create_segmenter
from src.normalization import (
    IrisNormalizer, 
    NormalizationResult, 
    IrisPreprocessingPipeline,
    create_normalizer,
    create_preprocessing_pipeline
)
from src.preprocessing import ImagePreprocessor, create_preprocessor
from src.feature_extraction import FeatureExtractor, create_feature_extractor
from src.classifiers import CrossValidator, ClassifierFactory, ClassificationResult
from src.metrics import MetricsCalculator, MetricsResult, format_metrics_table, format_confusion_matrix
from src.local_analysis import LocalAnalysisPipeline, SectorAnalysisResult, GridAnalysisResult


class IridologyPipeline:
    """
    Pipeline principal de análise de iridologia.
    
    Implementa o fluxo completo de:
    1. Segmentação de íris e pupila (opcional, para imagens raw)
    2. Normalização geométrica rubber sheet (opcional)
    3. Carregamento de dados
    4. Pré-processamento com transformações fotométricas
    5. Extração de features
    6. Treinamento e validação cruzada
    7. Avaliação com métricas completas
    8. Análises locais por região
    9. Testes de robustez
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Inicializa o pipeline.
        
        Args:
            config: Configuração do experimento
        """
        self.config = config or get_default_config()
        
        # Inicializa componentes de segmentação e normalização
        self.segmenter = create_segmenter(self.config.segmentation)
        self.normalizer = create_normalizer(self.config.normalization)
        self.preprocessing_pipeline = create_preprocessing_pipeline(
            self.config.segmentation,
            self.config.normalization
        )
        
        # Inicializa componentes de análise
        self.preprocessor = create_preprocessor(self.config.preprocessing)
        self.feature_extractor = create_feature_extractor(self.config.features)
        self.validator = CrossValidator(self.config.classifier)
        self.metrics_calculator = MetricsCalculator(self.config.metrics)
        self.local_analyzer = LocalAnalysisPipeline(
            self.config.local_analysis,
            self.config.classifier,
            self.config.features
        )
        
        # Cache de dados carregados
        self._data_cache = {}
        
        # Resultados
        self.results = {}
    
    # =========================================================================
    # Métodos de Segmentação e Normalização (para imagens raw)
    # =========================================================================
    
    def process_raw_images(self, images_path: str, 
                           output_path: Optional[str] = None,
                           labels_file: Optional[str] = None,
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processa imagens raw: segmentação + normalização.
        
        Esta função implementa as etapas 3 e 4 do pipeline do artigo:
        - Segmentação e delimitação anatômica
        - Normalização geométrica (rubber sheet)
        
        Args:
            images_path: Caminho para pasta com imagens (subpastas por classe)
            output_path: Caminho para salvar imagens processadas
            labels_file: Arquivo JSON com mapeamento imagem->label
            verbose: Se True, mostra progresso
            
        Returns:
            Tupla (normalized_images, labels)
        """
        import cv2
        import glob
        
        if verbose:
            print("=" * 60)
            print("PROCESSAMENTO DE IMAGENS RAW")
            print("=" * 60)
        
        all_images = []
        all_labels = []
        success_count = 0
        fail_count = 0
        
        # Carrega labels se fornecido
        label_map = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                label_map = json.load(f)
        
        # Procura imagens em subpastas (assumindo estrutura: control/, diabetic/)
        for class_name, label in [('control', 0), ('Control', 0), 
                                   ('diabetic', 1), ('Diabetic', 1),
                                   ('DM2', 1), ('healthy', 0)]:
            class_path = os.path.join(images_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            if verbose:
                print(f"\nProcessando classe: {class_name} (label={label})")
            
            # Encontra imagens
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            for pattern in patterns:
                image_files.extend(glob.glob(os.path.join(class_path, pattern)))
                image_files.extend(glob.glob(os.path.join(class_path, pattern.upper())))
            
            if verbose:
                print(f"  Encontradas {len(image_files)} imagens")
            
            for img_path in image_files:
                try:
                    # Carrega imagem
                    img = cv2.imread(img_path)
                    if img is None:
                        if verbose:
                            print(f"  Erro ao carregar: {img_path}")
                        fail_count += 1
                        continue
                    
                    # Segmenta
                    seg_result = self.segmenter.segment(img)
                    
                    if not seg_result.success:
                        if verbose:
                            print(f"  Falha na segmentação: {os.path.basename(img_path)}")
                        fail_count += 1
                        continue
                    
                    # Normaliza
                    norm_result = self.normalizer.normalize(img, seg_result)
                    
                    if not norm_result.success:
                        if verbose:
                            print(f"  Falha na normalização: {os.path.basename(img_path)}")
                        fail_count += 1
                        continue
                    
                    all_images.append(norm_result.normalized_image)
                    all_labels.append(label)
                    success_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"  Erro em {os.path.basename(img_path)}: {e}")
                    fail_count += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESUMO DO PROCESSAMENTO")
            print(f"  Sucesso: {success_count}")
            print(f"  Falhas:  {fail_count}")
            print(f"{'='*60}")
        
        # Converte para arrays
        if all_images:
            normalized_array = np.array(all_images)
            labels_array = np.array(all_labels)
            
            # Salva se solicitado
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                
                # Separa por classe
                control_images = normalized_array[labels_array == 0]
                diabetic_images = normalized_array[labels_array == 1]
                
                # Salva como pickle (compatível com formato existente)
                with open(os.path.join(output_path, 'controlImageArr.p'), 'wb') as f:
                    pickle.dump(control_images, f)
                with open(os.path.join(output_path, 'diabeteImageArr.p'), 'wb') as f:
                    pickle.dump(diabetic_images, f)
                
                # Cria arquivo DR_type vazio (placeholder)
                with open(os.path.join(output_path, 'DR_type.p'), 'wb') as f:
                    pickle.dump(np.zeros(len(diabetic_images)), f)
                
                if verbose:
                    print(f"Imagens salvas em: {output_path}")
            
            return normalized_array, labels_array
        else:
            return np.array([]), np.array([])
    
    def segment_single_image(self, image_path: str,
                              visualize: bool = True) -> SegmentationResult:
        """
        Segmenta uma única imagem (para teste/debug).
        
        Args:
            image_path: Caminho para a imagem
            visualize: Se True, exibe visualização
            
        Returns:
            SegmentationResult
        """
        import cv2
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        result = self.segmenter.segment(img)
        
        if visualize and result.success:
            self.segmenter.visualize(img, result)
        
        return result
    
    def normalize_single_image(self, image_path: str,
                                visualize: bool = True) -> Tuple[SegmentationResult, NormalizationResult]:
        """
        Processa (segmenta + normaliza) uma única imagem.
        
        Args:
            image_path: Caminho para a imagem
            visualize: Se True, exibe visualização
            
        Returns:
            Tupla (SegmentationResult, NormalizationResult)
        """
        import cv2
        import matplotlib.pyplot as plt
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        seg_result, norm_result = self.preprocessing_pipeline.process_single(img)
        
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Segmentação
            vis_img = self.segmenter.visualize(img, seg_result, show=False)
            axes[1].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Segmentação: {seg_result.message}')
            axes[1].axis('off')
            
            # Normalizada
            if norm_result.success:
                axes[2].imshow(norm_result.normalized_image, cmap='gray')
                axes[2].set_title('Normalizada (Rubber Sheet)')
            else:
                axes[2].text(0.5, 0.5, f'Erro: {norm_result.message}',
                            ha='center', va='center')
                axes[2].set_title('Normalização falhou')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return seg_result, norm_result
    
    def load_dataset(self, dataset_type: DatasetType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega um dataset específico.
        
        Args:
            dataset_type: Tipo de dataset a carregar
            
        Returns:
            Tupla (control_images, diabetic_images, dr_labels)
        """
        cache_key = dataset_type.value
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        dataset_path = self.config.data.get_dataset_path(dataset_type)
        
        print(f"Carregando dataset: {dataset_type.value} de {dataset_path}")
        
        # Carrega arquivos pickle
        control_path = os.path.join(dataset_path, self.config.data.control_file)
        diabetic_path = os.path.join(dataset_path, self.config.data.diabetic_file)
        dr_type_path = os.path.join(dataset_path, self.config.data.dr_type_file)
        
        with open(control_path, 'rb') as f:
            control_images = pickle.load(f, encoding='latin1')
        
        with open(diabetic_path, 'rb') as f:
            diabetic_images = pickle.load(f, encoding='latin1')
        
        with open(dr_type_path, 'rb') as f:
            dr_labels = pickle.load(f, encoding='latin1')
        
        print(f"  Control: {control_images.shape}")
        print(f"  Diabetic: {diabetic_images.shape}")
        
        self._data_cache[cache_key] = (control_images, diabetic_images, dr_labels)
        
        return control_images, diabetic_images, dr_labels
    
    def prepare_data(self, dataset_type: DatasetType,
                     channel: ColorChannel = ColorChannel.GRAY,
                     transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados para treinamento.
        
        Args:
            dataset_type: Tipo de dataset
            channel: Canal de cor
            transform: Transformação fotométrica
            
        Returns:
            Tupla (features, labels)
        """
        control_images, diabetic_images, _ = self.load_dataset(dataset_type)
        
        n_diabetic = diabetic_images.shape[0]
        n_control = control_images.shape[0]
        
        all_features = []
        all_labels = []
        
        print(f"Extraindo features ({transform.value}, {channel.value})...")
        
        # Processa imagens diabéticas
        for i in range(n_diabetic):
            img = np.squeeze(diabetic_images[i])
            
            # Pré-processa
            processed = self.preprocessor.preprocess(img, channel, transform)
            
            # Extrai features
            features = self.feature_extractor.extract_features(processed)
            all_features.append(features)
            all_labels.append(1)  # Diabético
        
        # Processa imagens controle
        for i in range(n_control):
            img = np.squeeze(control_images[i])
            
            # Pré-processa
            processed = self.preprocessor.preprocess(img, channel, transform)
            
            # Extrai features
            features = self.feature_extractor.extract_features(processed)
            all_features.append(features)
            all_labels.append(0)  # Controle
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"  Shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def run_classification(self, dataset_type: DatasetType,
                            channel: ColorChannel = ColorChannel.GRAY,
                            transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                            ) -> Dict[str, ClassificationResult]:
        """
        Executa classificação completa.
        
        Args:
            dataset_type: Tipo de dataset
            channel: Canal de cor
            transform: Transformação fotométrica
            
        Returns:
            Dicionário de resultados por classificador
        """
        X, y = self.prepare_data(dataset_type, channel, transform)
        
        print(f"\nValidação cruzada ({self.config.classifier.n_folds} folds)...")
        
        results = self.validator.validate_all(X, y, dataset_type.value)
        
        return results
    
    def run_all_datasets(self, 
                          channel: ColorChannel = ColorChannel.GRAY,
                          transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                          ) -> Dict[str, Dict[str, ClassificationResult]]:
        """
        Executa classificação em todos os datasets.
        
        Args:
            channel: Canal de cor
            transform: Transformação fotométrica
            
        Returns:
            Dicionário de resultados por dataset e classificador
        """
        all_results = {}
        
        for dataset_type in self.config.data.datasets:
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset_type.value}")
            print(f"{'='*60}")
            
            try:
                results = self.run_classification(dataset_type, channel, transform)
                all_results[dataset_type.value] = results
            except Exception as e:
                print(f"Erro ao processar {dataset_type.value}: {e}")
                continue
        
        self.results['all_datasets'] = all_results
        return all_results
    
    def run_robustness_test(self, 
                             dataset_type: DatasetType = DatasetType.PERSON_BASE
                             ) -> Dict[str, Dict[str, ClassificationResult]]:
        """
        Executa teste de robustez com transformações fotométricas.
        
        Args:
            dataset_type: Dataset a usar
            
        Returns:
            Dicionário de resultados por transformação
        """
        robustness_results = {}
        
        print(f"\n{'='*60}")
        print("TESTE DE ROBUSTEZ FOTOMÉTRICA")
        print(f"Dataset: {dataset_type.value}")
        print(f"{'='*60}")
        
        for transform in self.config.preprocessing.transforms:
            print(f"\n--- Transformação: {transform.value} ---")
            
            results = self.run_classification(
                dataset_type, 
                ColorChannel.GRAY, 
                transform
            )
            robustness_results[transform.value] = results
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def run_local_analysis(self, 
                            dataset_type: DatasetType = DatasetType.PERSON_BASE,
                            classifier_type: ClassifierType = ClassifierType.LR,
                            eye_side: str = 'left'
                            ) -> Dict[str, any]:
        """
        Executa análise local por região.
        
        Args:
            dataset_type: Dataset a usar
            classifier_type: Classificador a usar
            eye_side: Lado do olho
            
        Returns:
            Dicionário com resultados de análise local
        """
        control_images, diabetic_images, _ = self.load_dataset(dataset_type)
        
        # Prepara imagens normalizadas
        all_images = []
        all_labels = []
        
        n_diabetic = diabetic_images.shape[0]
        n_control = control_images.shape[0]
        
        for i in range(n_diabetic):
            img = np.squeeze(diabetic_images[i])
            processed = self.preprocessor.preprocess(img)
            all_images.append(processed)
            all_labels.append(1)
        
        for i in range(n_control):
            img = np.squeeze(control_images[i])
            processed = self.preprocessor.preprocess(img)
            all_images.append(processed)
            all_labels.append(0)
        
        images = np.array(all_images)
        labels = np.array(all_labels)
        
        # Executa análise local
        local_results = self.local_analyzer.run_full_analysis(
            images, labels, classifier_type, eye_side, dataset_type.value
        )
        
        self.results['local_analysis'] = local_results
        return local_results
    
    def generate_report(self) -> str:
        """
        Gera relatório completo dos resultados.
        
        Returns:
            String com relatório formatado
        """
        lines = []
        
        lines.append("=" * 70)
        lines.append("RELATÓRIO DE ANÁLISE DE IRIDOLOGIA")
        lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # Resultados por dataset
        if 'all_datasets' in self.results:
            lines.append("\n" + "=" * 50)
            lines.append("RESULTADOS POR DATASET")
            lines.append("=" * 50)
            
            for dataset_name, clf_results in self.results['all_datasets'].items():
                lines.append(f"\n--- {dataset_name} ---")
                
                for clf_name, result in clf_results.items():
                    lines.append(
                        f"{clf_name}: Acc={result.accuracy_mean:.4f}±{result.accuracy_std:.3f}, "
                        f"Sens={result.sensitivity_mean:.4f}, Spec={result.specificity_mean:.4f}"
                    )
        
        # Resultados de robustez
        if 'robustness' in self.results:
            lines.append("\n" + "=" * 50)
            lines.append("TESTE DE ROBUSTEZ FOTOMÉTRICA")
            lines.append("=" * 50)
            
            for transform_name, clf_results in self.results['robustness'].items():
                lines.append(f"\n--- {transform_name} ---")
                
                for clf_name, result in clf_results.items():
                    lines.append(
                        f"{clf_name}: Acc={result.accuracy_mean:.4f}±{result.accuracy_std:.3f}"
                    )
        
        # Resultados de análise local
        if 'local_analysis' in self.results:
            lines.append("\n" + "=" * 50)
            lines.append("ANÁLISE LOCAL")
            lines.append("=" * 50)
            
            if 'sector_analysis' in self.results['local_analysis']:
                sector = self.results['local_analysis']['sector_analysis']
                best = sector.get_best_sector()
                lines.append(f"Melhor setor: {best[0]}°-{best[1]}° (Acc={best[2]:.4f})")
            
            if 'pancreatic_analysis' in self.results['local_analysis']:
                pancreatic = self.results['local_analysis']['pancreatic_analysis']
                best = pancreatic.get_best_cell()
                lines.append(f"Melhor célula pancreática: ({best[0]},{best[1]}) (Acc={best[2]:.4f})")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: Optional[str] = None):
        """
        Salva resultados em arquivos.
        
        Args:
            output_dir: Diretório de saída
        """
        output_dir = output_dir or self.config.data.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva relatório texto
        report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"Relatório salvo em: {report_path}")
        
        # Salva resultados em JSON
        results_dict = {}
        
        if 'all_datasets' in self.results:
            results_dict['all_datasets'] = {}
            for dataset, clf_results in self.results['all_datasets'].items():
                results_dict['all_datasets'][dataset] = {
                    clf: result.to_dict() for clf, result in clf_results.items()
                }
        
        if 'robustness' in self.results:
            results_dict['robustness'] = {}
            for transform, clf_results in self.results['robustness'].items():
                results_dict['robustness'][transform] = {
                    clf: result.to_dict() for clf, result in clf_results.items()
                }
        
        json_path = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Resultados JSON salvos em: {json_path}")


def create_accuracy_table(results: Dict[str, Dict[str, ClassificationResult]]) -> str:
    """
    Cria tabela de acurácia no formato do artigo (Tabela 1).
    
    Args:
        results: Resultados por dataset e classificador
        
    Returns:
        String formatada como tabela
    """
    classifiers = ['AdaBoost', 'MLP', 'RandomForest', 'SVM', 'LogisticRegression']
    datasets = ['all', 'L_split', 'R_split', 'personBase', 'personBase_invert']
    
    lines = []
    lines.append("\nTabela: Acurácia por dataset e classificador (pixel features)")
    lines.append("-" * 80)
    
    # Header
    header = f"{'Dataset':<20}"
    for clf in classifiers:
        header += f"{clf[:8]:<12}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Rows
    for dataset in datasets:
        if dataset in results:
            row = f"{dataset:<20}"
            for clf in classifiers:
                if clf in results[dataset]:
                    acc = results[dataset][clf].accuracy_mean
                    row += f"{acc:.4f}      "
                else:
                    row += f"{'N/A':<12}"
            lines.append(row)
    
    lines.append("-" * 80)
    
    return "\n".join(lines)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Pipeline de Análise de Iridologia para DM2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Análise com dados pré-processados (pickle)
  python main.py --dataset personBase
  python main.py --run-all
  python main.py --robustness-test
  python main.py --local-analysis --eye-side left

  # Processamento de imagens raw (segmentação + normalização)
  python main.py --process-raw --raw-path ./raw_images --output-processed ./processed
  python main.py --segment-single --image-path ./image.jpg
  
  # Pipeline completo: raw -> classificação
  python main.py --full-pipeline --raw-path ./raw_images
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='personBase',
        choices=['all', 'L_split', 'R_split', 'personBase', 'personBase_invert'],
        help='Dataset a usar'
    )
    
    parser.add_argument(
        '--channel',
        type=str,
        default='gray',
        choices=['gray', 'h', 's', 'v', 'r', 'g', 'b'],
        help='Canal de cor'
    )
    
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Executa em todos os datasets'
    )
    
    parser.add_argument(
        '--robustness-test',
        action='store_true',
        help='Executa teste de robustez fotométrica'
    )
    
    parser.add_argument(
        '--local-analysis',
        action='store_true',
        help='Executa análise local por região'
    )
    
    parser.add_argument(
        '--eye-side',
        type=str,
        default='left',
        choices=['left', 'right'],
        help='Lado do olho para análise local'
    )
    
    # Novos argumentos para processamento de imagens raw
    parser.add_argument(
        '--process-raw',
        action='store_true',
        help='Processa imagens raw (segmentação + normalização)'
    )
    
    parser.add_argument(
        '--raw-path',
        type=str,
        default=None,
        help='Caminho para imagens raw (estrutura: control/, diabetic/)'
    )
    
    parser.add_argument(
        '--output-processed',
        type=str,
        default='Data/Processed',
        help='Caminho para salvar imagens processadas'
    )
    
    parser.add_argument(
        '--segment-single',
        action='store_true',
        help='Segmenta uma única imagem (para teste)'
    )
    
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Caminho para imagem única (usado com --segment-single)'
    )
    
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Executa pipeline completo: raw -> processamento -> classificação'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Diretório para salvar resultados'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='Data/Data',
        help='Caminho para os dados processados'
    )
    
    args = parser.parse_args()
    
    # Cria configuração
    config = get_default_config()
    config.data.data_root = args.data_path
    config.data.output_dir = args.output_dir
    
    # Inicializa pipeline
    pipeline = IridologyPipeline(config)
    
    # Mapeia argumentos
    dataset_map = {
        'all': DatasetType.ALL,
        'L_split': DatasetType.L_SPLIT,
        'R_split': DatasetType.R_SPLIT,
        'personBase': DatasetType.PERSON_BASE,
        'personBase_invert': DatasetType.PERSON_BASE_INVERT
    }
    
    channel_map = {
        'gray': ColorChannel.GRAY,
        'h': ColorChannel.H,
        's': ColorChannel.S,
        'v': ColorChannel.V,
        'r': ColorChannel.R,
        'g': ColorChannel.G,
        'b': ColorChannel.B
    }
    
    # Executa
    try:
        # Processamento de imagem única
        if args.segment_single:
            if not args.image_path:
                print("Erro: --image-path é obrigatório com --segment-single")
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print("SEGMENTAÇÃO DE IMAGEM ÚNICA")
            print("=" * 60)
            
            seg_result, norm_result = pipeline.normalize_single_image(
                args.image_path, visualize=True
            )
            
            print(f"\nResultado da segmentação:")
            print(f"  Sucesso: {seg_result.success}")
            print(f"  Mensagem: {seg_result.message}")
            print(f"  Centro pupila: {seg_result.pupil_center}")
            print(f"  Raio pupila: {seg_result.pupil_radius}")
            print(f"  Centro íris: {seg_result.iris_center}")
            print(f"  Raio íris: {seg_result.iris_radius}")
            
            if norm_result.success:
                print(f"\nNormalização:")
                print(f"  Shape: {norm_result.normalized_image.shape}")
                print(f"  Resolução angular: {norm_result.angular_resolution:.2f}°/pixel")
        
        # Processamento de imagens raw em lote
        elif args.process_raw:
            if not args.raw_path:
                print("Erro: --raw-path é obrigatório com --process-raw")
                sys.exit(1)
            
            normalized, labels = pipeline.process_raw_images(
                args.raw_path,
                args.output_processed,
                verbose=True
            )
            
            print(f"\nImagens processadas: {len(normalized)}")
            print(f"  Controle: {np.sum(labels == 0)}")
            print(f"  Diabético: {np.sum(labels == 1)}")
        
        # Pipeline completo: raw -> classificação
        elif args.full_pipeline:
            if not args.raw_path:
                print("Erro: --raw-path é obrigatório com --full-pipeline")
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETO: RAW -> CLASSIFICAÇÃO")
            print("=" * 60)
            
            # Etapa 1: Processar imagens raw
            print("\n[ETAPA 1/3] Processamento de imagens raw...")
            normalized, labels = pipeline.process_raw_images(
                args.raw_path,
                args.output_processed,
                verbose=True
            )
            
            if len(normalized) == 0:
                print("Erro: Nenhuma imagem foi processada com sucesso")
                sys.exit(1)
            
            # Etapa 2: Extração de features
            print("\n[ETAPA 2/3] Extração de features...")
            all_features = []
            for img in normalized:
                if len(img.shape) == 3:
                    img = img[:, :, 0]  # Usa primeiro canal
                features = pipeline.feature_extractor.extract_features(img)
                all_features.append(features)
            
            X = np.array(all_features, dtype=np.float32)
            y = labels.astype(np.int32)
            
            print(f"  Features shape: {X.shape}")
            
            # Etapa 3: Classificação
            print("\n[ETAPA 3/3] Classificação...")
            results = pipeline.validator.validate_all(X, y, "processed_raw")
            
            # Imprime resultados
            print("\n" + "=" * 60)
            print("RESULTADOS")
            print("=" * 60)
            for clf_name, result in results.items():
                print(f"  {clf_name}:")
                print(f"    Accuracy: {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}")
                print(f"    Sensitivity: {result.sensitivity_mean:.4f}")
                print(f"    Specificity: {result.specificity_mean:.4f}")
                print(f"    F1-score: {result.f1_mean:.4f}")
        
        # Análise com dados já processados
        elif args.run_all:
            print("\n" + "=" * 60)
            print("EXECUTANDO ANÁLISE EM TODOS OS DATASETS")
            print("=" * 60)
            results = pipeline.run_all_datasets(channel_map[args.channel])
            
            # Imprime tabela de acurácia
            print(create_accuracy_table(results))
        
        elif args.robustness_test:
            print("\n" + "=" * 60)
            print("EXECUTANDO TESTE DE ROBUSTEZ")
            print("=" * 60)
            pipeline.run_robustness_test(dataset_map[args.dataset])
        
        elif args.local_analysis:
            print("\n" + "=" * 60)
            print("EXECUTANDO ANÁLISE LOCAL")
            print("=" * 60)
            pipeline.run_local_analysis(
                dataset_map[args.dataset],
                ClassifierType.LR,
                args.eye_side
            )
        
        else:
            print("\n" + "=" * 60)
            print(f"EXECUTANDO ANÁLISE NO DATASET: {args.dataset}")
            print("=" * 60)
            results = pipeline.run_classification(
                dataset_map[args.dataset],
                channel_map[args.channel]
            )
            
            # Imprime resultados
            print("\nResultados:")
            for clf_name, result in results.items():
                print(f"  {clf_name}:")
                print(f"    Accuracy: {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}")
                print(f"    Sensitivity: {result.sensitivity_mean:.4f}")
                print(f"    Specificity: {result.specificity_mean:.4f}")
                print(f"    F1-score: {result.f1_mean:.4f}")
        
        # Salva resultados (exceto para operações de teste)
        if config.save_results and not args.segment_single:
            pipeline.save_results()
        
        # Imprime relatório
        if not args.segment_single and not args.process_raw:
            print(pipeline.generate_report())
        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        print("Verifique se o caminho dos dados está correto.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
