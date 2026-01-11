#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BENCHMARK LEADERBOARD: Prescriptive Process Monitoring
-------------------------------------------------------
Este script genera un leaderboard completo estilo LLM benchmarks para evaluar
todos los modelos causales de prescriptive process monitoring.

Características:
- Múltiples datasets (BPI 2012, 2017, 2019, etc.)
- Múltiples runs con diferentes seeds (para estadísticas robustas)
- Intervalos de confianza y errores estándar
- Múltiples objetivos (Net Gain, Time Reduction, Cost Reduction)
- Comparación justa (mismo split, mismo preprocesamiento)
- Reporte en formato tabla markdown/CSV
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Agregar directorios al path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.logger_utils import setup_logger
from benchmark_evaluator import BenchmarkEvaluator

logger = setup_logger(__name__)


@dataclass
class ModelResult:
    """Resultado de un modelo en un run específico"""
    model_name: str
    dataset: str
    run_id: int
    net_gain: float
    lift_vs_bau: float
    intervention_percentage: float
    violation_percentage: float
    auc_qini: Optional[float]
    latency_ms: Optional[float]
    training_complexity: str
    time_reduction: Optional[float] = None
    cost_reduction: Optional[float] = None


@dataclass
class ModelStats:
    """Estadísticas agregadas de un modelo (promedio sobre múltiples runs)"""
    model_name: str
    dataset: str
    n_runs: int
    net_gain_mean: float
    net_gain_std: float
    net_gain_ci_lower: float
    net_gain_ci_upper: float
    lift_vs_bau_mean: float
    lift_vs_bau_std: float
    intervention_percentage_mean: float
    violation_percentage_mean: float
    auc_qini_mean: Optional[float]
    latency_ms_mean: Optional[float]
    training_complexity: str


class BenchmarkLeaderboard:
    """
    Genera un leaderboard completo para modelos de prescriptive process monitoring.
    
    Similar a los benchmarks de LLMs (GLUE, SuperGLUE, MMLU), pero adaptado para
    prescriptive process monitoring.
    """
    
    def __init__(
        self,
        datasets: List[str],
        n_runs: int = 5,
        random_seeds: Optional[List[int]] = None
    ):
        """
        Args:
            datasets: Lista de nombres de datasets a evaluar
            n_runs: Número de runs con diferentes seeds
            random_seeds: Lista de seeds a usar (si None, genera automáticamente)
        """
        self.datasets = datasets
        self.n_runs = n_runs
        if random_seeds is None:
            self.random_seeds = list(range(42, 42 + n_runs))
        else:
            self.random_seeds = random_seeds[:n_runs]
        
        self.results: List[ModelResult] = []
        self.stats: List[ModelStats] = []
    
    def add_result(self, result: ModelResult):
        """Agrega un resultado al leaderboard"""
        self.results.append(result)
    
    def compute_statistics(self) -> pd.DataFrame:
        """
        Calcula estadísticas agregadas (promedio, std, CI) para cada modelo.
        
        Returns:
            DataFrame con estadísticas agregadas
        """
        if not self.results:
            logger.warning("No hay resultados para calcular estadísticas")
            return pd.DataFrame()
        
        df_results = pd.DataFrame([
            {
                'model_name': r.model_name,
                'dataset': r.dataset,
                'run_id': r.run_id,
                'net_gain': r.net_gain,
                'lift_vs_bau': r.lift_vs_bau,
                'intervention_percentage': r.intervention_percentage,
                'violation_percentage': r.violation_percentage,
                'auc_qini': r.auc_qini,
                'latency_ms': r.latency_ms,
                'training_complexity': r.training_complexity
            }
            for r in self.results
        ])
        
        # Agrupar por modelo y dataset
        stats_list = []
        for (model_name, dataset), group in df_results.groupby(['model_name', 'dataset']):
            n_runs = len(group)
            
            # Calcular estadísticas para net_gain
            net_gain_mean = group['net_gain'].mean()
            net_gain_std = group['net_gain'].std()
            # CI 95% usando t-distribution
            from scipy import stats
            if n_runs > 1:
                t_critical = stats.t.ppf(0.975, df=n_runs-1)
                se = net_gain_std / np.sqrt(n_runs)
                net_gain_ci_lower = net_gain_mean - t_critical * se
                net_gain_ci_upper = net_gain_mean + t_critical * se
            else:
                net_gain_ci_lower = net_gain_mean
                net_gain_ci_upper = net_gain_mean
            
            # Otras métricas
            lift_mean = group['lift_vs_bau'].mean()
            lift_std = group['lift_vs_bau'].std()
            intervention_mean = group['intervention_percentage'].mean()
            violation_mean = group['violation_percentage'].mean()
            auc_qini_mean = group['auc_qini'].mean() if not group['auc_qini'].isna().all() else None
            latency_mean = group['latency_ms'].mean() if not group['latency_ms'].isna().all() else None
            complexity = group['training_complexity'].iloc[0]  # Debería ser igual para todos
            
            stats_list.append(ModelStats(
                model_name=model_name,
                dataset=dataset,
                n_runs=n_runs,
                net_gain_mean=net_gain_mean,
                net_gain_std=net_gain_std,
                net_gain_ci_lower=net_gain_ci_lower,
                net_gain_ci_upper=net_gain_ci_upper,
                lift_vs_bau_mean=lift_mean,
                lift_vs_bau_std=lift_std,
                intervention_percentage_mean=intervention_mean,
                violation_percentage_mean=violation_mean,
                auc_qini_mean=auc_qini_mean,
                latency_ms_mean=latency_mean,
                training_complexity=complexity
            ))
        
        self.stats = stats_list
        
        # Convertir a DataFrame
        df_stats = pd.DataFrame([
            {
                'Model': s.model_name,
                'Dataset': s.dataset,
                'N Runs': s.n_runs,
                'Net Gain ($)': f"{s.net_gain_mean:.2f} ± {s.net_gain_std:.2f}",
                'Net Gain CI 95%': f"[{s.net_gain_ci_lower:.2f}, {s.net_gain_ci_upper:.2f}]",
                'Lift vs BAU (%)': f"{s.lift_vs_bau_mean:.2f} ± {s.lift_vs_bau_std:.2f}",
                '% Intervenciones': f"{s.intervention_percentage_mean:.2f}",
                '% Violaciones': f"{s.violation_percentage_mean:.2f}",
                'AUC-Qini': f"{s.auc_qini_mean:.4f}" if s.auc_qini_mean is not None else "N/A",
                'Latencia (ms)': f"{s.latency_ms_mean:.2f}" if s.latency_ms_mean is not None else "N/A",
                'Complejidad': s.training_complexity
            }
            for s in self.stats
        ])
        
        return df_stats
    
    def generate_markdown_table(self, df_stats: pd.DataFrame) -> str:
        """
        Genera una tabla markdown estilo leaderboard de LLMs.
        
        Returns:
            String con tabla markdown
        """
        if df_stats.empty:
            return "No hay resultados para mostrar"
        
        # Ordenar por Net Gain descendente
        df_sorted = df_stats.copy()
        
        # Extraer valor numérico de Net Gain para ordenar
        df_sorted['_sort_key'] = df_sorted['Net Gain ($)'].str.extract(r'([\d.]+)').astype(float)
        df_sorted = df_sorted.sort_values('_sort_key', ascending=False)
        df_sorted = df_sorted.drop('_sort_key', axis=1)
        
        # Generar tabla markdown
        lines = []
        lines.append("| Paper (Modelo) | Dataset | 💰 Net Gain ($) (OPE-IPW) | 📈 Lift vs BAU | 📉 % Intervenciones | 🛡️ % Violación | 🎯 AUC-Qini | 🐢 Latencia (ms) | 🧠 Complejidad Entrenamiento |")
        lines.append("|" + "|".join(["---"] * 9) + "|")
        
        for _, row in df_sorted.iterrows():
            lines.append(
                f"| {row['Model']} | {row['Dataset']} | {row['Net Gain ($)']} | "
                f"{row['Lift vs BAU (%)']} | {row['% Intervenciones']} | "
                f"{row['% Violaciones']} | {row['AUC-Qini']} | {row['Latencia (ms)']} | "
                f"{row['Complejidad']} |"
            )
        
        return "\n".join(lines)
    
    def save_results(
        self,
        output_dir: str,
        df_stats: pd.DataFrame,
        markdown_table: str
    ):
        """Guarda resultados en múltiples formatos"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV con estadísticas
        csv_path = output_path / "leaderboard_stats.csv"
        df_stats_clean = df_stats.copy()
        # Separar Net Gain en columnas numéricas para CSV
        df_stats_clean['Net_Gain_Mean'] = df_stats_clean['Net Gain ($)'].str.extract(r'([\d.]+)').astype(float)
        df_stats_clean['Net_Gain_Std'] = df_stats_clean['Net Gain ($)'].str.extract(r'± ([\d.]+)').astype(float)
        df_stats_clean.to_csv(csv_path, index=False)
        logger.info(f"✅ Estadísticas guardadas en: {csv_path}")
        
        # Guardar tabla markdown
        md_path = output_path / "leaderboard.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Prescriptive Process Monitoring Benchmark Leaderboard\n\n")
            f.write("## Métricas\n\n")
            f.write(markdown_table)
            f.write("\n\n## Notas\n\n")
            f.write("- **Net Gain**: Valor esperado de la política usando IPW (OPE-IPW)\n")
            f.write("- **Lift vs BAU**: Mejora porcentual respecto al Business As Usual\n")
            f.write("- **% Intervenciones**: Porcentaje de casos donde el modelo recomienda intervenir\n")
            f.write("- **% Violaciones**: Porcentaje de casos donde la recomendación viola reglas de negocio\n")
            f.write("- **AUC-Qini**: Área bajo la curva Qini (calidad del ranking de uplift)\n")
            f.write("- **Latencia**: Tiempo de inferencia en milisegundos\n")
            f.write("- **Complejidad**: Complejidad computacional del entrenamiento\n")
        logger.info(f"✅ Tabla markdown guardada en: {md_path}")
        
        # Guardar resultados raw (todos los runs)
        raw_path = output_path / "raw_results.json"
        raw_data = [
            {
                'model_name': r.model_name,
                'dataset': r.dataset,
                'run_id': r.run_id,
                'net_gain': r.net_gain,
                'lift_vs_bau': r.lift_vs_bau,
                'intervention_percentage': r.intervention_percentage,
                'violation_percentage': r.violation_percentage,
                'auc_qini': r.auc_qini,
                'latency_ms': r.latency_ms,
                'training_complexity': r.training_complexity
            }
            for r in self.results
        ]
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        logger.info(f"✅ Resultados raw guardados en: {raw_path}")


def main():
    """
    Ejemplo de uso del Benchmark Leaderboard.
    
    Este script debería ser llamado desde otros scripts de evaluación
    que ejecuten los modelos y agreguen resultados.
    """
    logger.info("="*80)
    logger.info("BENCHMARK LEADERBOARD: Prescriptive Process Monitoring")
    logger.info("="*80)
    
    # Crear leaderboard
    leaderboard = BenchmarkLeaderboard(
        datasets=['BPI2017', 'BPI2012', 'BPI2019'],  # Ejemplo
        n_runs=5
    )
    
    # Ejemplo: agregar resultados (normalmente vendrían de scripts de evaluación)
    # leaderboard.add_result(ModelResult(...))
    
    # Calcular estadísticas
    df_stats = leaderboard.compute_statistics()
    
    # Generar tabla markdown
    markdown_table = leaderboard.generate_markdown_table(df_stats)
    
    # Guardar resultados
    output_dir = os.path.join(project_root, "results/benchmark/leaderboard")
    leaderboard.save_results(output_dir, df_stats, markdown_table)
    
    print("\n" + markdown_table)


if __name__ == "__main__":
    main()

