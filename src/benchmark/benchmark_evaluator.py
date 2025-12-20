#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Evaluator para Prescriptive Process Monitoring.

Calcula métricas de rendimiento para comparar diferentes algoritmos
(RL, Causal Inference, etc.) sobre el dataset BPI Challenge 2017.
"""

import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import roc_auc_score

# Agregar src/ al PYTHONPATH para encontrar utils
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)  # src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.logger_utils import setup_logger
from utils.config import load_config

# Configurar logger
logger = setup_logger(__name__)

# Constantes del Negocio (BPI 2017) - Valores por defecto
REWARD_SUCCESS = 100.0  # Ganancia si el préstamo fue aceptado
COST_INTERVENTION = 20.0  # Costo si se llama (intervención)
COST_TIME_DAY = 1.0  # Costo por día de duración


def load_benchmark_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Carga la configuración del benchmark desde benchmark_config.yaml.
    
    Args:
        config_path: Ruta al archivo de configuración. Si es None, busca configs/benchmark_config.yaml
                    relativo al directorio del proyecto.
    
    Returns:
        Diccionario con la configuración cargada, o None si hay error.
    """
    if config_path is None:
        # Determinar directorio base del proyecto
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # script_dir = src/benchmark/
        src_dir = os.path.dirname(script_dir)  # src/
        project_root = os.path.dirname(src_dir)  # proyecto raíz
        config_path = os.path.join(project_root, "configs", "benchmark_config.yaml")
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Error cargando configuración del benchmark: {e}")
        return None


def get_default_constants() -> Tuple[float, float, float]:
    """
    Obtiene las constantes del negocio desde benchmark_config.yaml o usa valores por defecto.
    
    Returns:
        Tupla (reward_success, cost_intervention, cost_time_day)
    """
    benchmark_config = load_benchmark_config()
    if benchmark_config:
        evaluator_config = benchmark_config.get("evaluator", {})
        reward = evaluator_config.get("reward_success", REWARD_SUCCESS)
        cost_int = evaluator_config.get("cost_intervention", COST_INTERVENTION)
        cost_time = evaluator_config.get("cost_time_day", COST_TIME_DAY)
        return reward, cost_int, cost_time
    return REWARD_SUCCESS, COST_INTERVENTION, COST_TIME_DAY


class BenchmarkEvaluator:
    """
    Evaluador de métricas para Prescriptive Process Monitoring.
    
    Calcula 7 métricas principales:
    1. Net Gain ($) - Valor esperado de la política usando IPW
    2. Lift vs BAU (%) - Mejora respecto al Business As Usual
    3. % Intervenciones - Eficiencia del modelo
    4. % Violación - Safety/Compliance
    5. AUC-Qini - Calidad del ranking de uplift
    6. Latencia (ms) - Tiempo de inferencia
    7. Complejidad Entrenamiento - Categoría cualitativa
    """
    
    def __init__(
        self,
        reward_success: Optional[float] = None,
        cost_intervention: Optional[float] = None,
        cost_time_day: Optional[float] = None
    ):
        """
        Inicializa el evaluador.
        
        Si los parámetros son None, intenta cargarlos desde benchmark_config.yaml.
        Si no están disponibles en el config, usa valores por defecto.
        
        Args:
            reward_success: Ganancia si el outcome es exitoso (None = cargar desde config)
            cost_intervention: Costo de una intervención (None = cargar desde config)
            cost_time_day: Costo por día de duración (None = cargar desde config)
        """
        # Cargar constantes desde config o usar valores por defecto
        default_reward, default_cost_int, default_cost_time = get_default_constants()
        
        self.reward_success = reward_success if reward_success is not None else default_reward
        self.cost_intervention = cost_intervention if cost_intervention is not None else default_cost_int
        self.cost_time_day = cost_time_day if cost_time_day is not None else default_cost_time
        
        logger.debug(f"BenchmarkEvaluator inicializado con: reward={self.reward_success}, "
                    f"cost_intervention={self.cost_intervention}, cost_time_day={self.cost_time_day}")
    
    def check_safety(self, case_row: pd.Series) -> bool:
        """
        Verifica si una acción recomendada viola reglas de negocio.
        
        Reglas BPI 2017:
        1. No llamar si el estado actual es "A_Cancelled" o "O_Refused"
        2. No llamar si ya se llamó en los últimos 2 días (frecuencia)
        
        Args:
            case_row: Fila del DataFrame con información del caso
        
        Returns:
            True si la acción es segura, False si viola reglas
        """
        # Si el modelo no recomienda intervenir, es seguro
        if case_row.get('action_model', 0) == 0:
            return True
        
        # Regla 1: No llamar si el estado es "A_Cancelled" o "O_Refused"
        current_state = case_row.get('current_state', '')
        if current_state in ['A_Cancelled', 'O_Refused']:
            return False
        
        # Regla 2: No llamar si ya se llamó en los últimos 2 días
        # (Asumiendo que tenemos información de intervenciones previas)
        days_since_last_intervention = case_row.get('days_since_last_intervention', 999)
        if days_since_last_intervention < 2:
            return False
        
        return True
    
    def calculate_observed_reward(
        self,
        outcome_observed: int,
        treatment_observed: int,
        duration_days: float
    ) -> float:
        """
        Calcula el reward observado para un caso.
        
        Args:
            outcome_observed: 1 si el préstamo fue aceptado, 0 si no
            treatment_observed: 1 si hubo intervención, 0 si no
            duration_days: Duración total del caso en días
        
        Returns:
            Reward observado
        """
        reward = outcome_observed * self.reward_success
        cost_intervention = treatment_observed * self.cost_intervention
        cost_time = duration_days * self.cost_time_day
        
        return reward - cost_intervention - cost_time
    
    def calculate_net_gain(
        self,
        df_results: pd.DataFrame
    ) -> float:
        """
        Calcula el Net Gain ($) usando Inverse Propensity Weighting (IPW).
        
        Para cada caso i:
        - Si action_model == treatment_observed: usar reward observado / propensity_score
        - Si no coinciden: usar 0 o estimación alternativa
        
        Nota: Para un estimador más robusto, se podría usar Doubly Robust (DR),
        pero IPW es suficiente para este benchmark.
        
        Args:
            df_results: DataFrame con columnas requeridas
        
        Returns:
            Net Gain promedio por caso
        """
        if 'propensity_score' not in df_results.columns:
            logger.warning("propensity_score no encontrado, usando 0.5 como fallback")
            df_results = df_results.copy()
            df_results['propensity_score'] = 0.5
        
        # Validar que propensity_score esté en rango válido
        if df_results['propensity_score'].min() <= 0 or df_results['propensity_score'].max() >= 1:
            logger.warning("Propensity scores fuera de rango (0,1), aplicando clipping")
            df_results = df_results.copy()
            df_results['propensity_score'] = df_results['propensity_score'].clip(0.01, 0.99)
        
        # Calcular rewards observados
        df_results = df_results.copy()
        df_results['observed_reward'] = df_results.apply(
            lambda row: self.calculate_observed_reward(
                row.get('outcome_observed', 0),
                row.get('treatment_observed', 0),
                row.get('duration_days', 0)
            ),
            axis=1
        )
        
        # Calcular rewards ajustados usando IPW
        # Solo usamos casos donde action_model == treatment_observed
        mask_match = df_results['action_model'] == df_results['treatment_observed']
        
        # Para casos que coinciden: reward / propensity_score (IPW estándar)
        # Para casos que no coinciden: 0 (no tenemos contrafactual directo)
        # Nota: En un estimador DR, aquí usaríamos un modelo de regresión para estimar E[Y|X,T]
        df_results['adjusted_reward'] = np.where(
            mask_match,
            df_results['observed_reward'] / df_results['propensity_score'],
            0.0
        )
        
        net_gain = df_results['adjusted_reward'].mean()
        
        # Estadísticas adicionales para debugging
        match_rate = mask_match.sum() / len(df_results)
        logger.info(f"Net Gain calculado: ${net_gain:.2f}")
        logger.info(f"  Casos con coincidencia: {mask_match.sum()}/{len(df_results)} ({match_rate:.1%})")
        logger.info(f"  Propensity score: min={df_results['propensity_score'].min():.3f}, "
                   f"max={df_results['propensity_score'].max():.3f}, "
                   f"mean={df_results['propensity_score'].mean():.3f}")
        
        return float(net_gain)
    
    def calculate_historical_net_gain(
        self,
        df_results: pd.DataFrame
    ) -> float:
        """
        Calcula el Net Gain histórico (Business As Usual).
        
        Args:
            df_results: DataFrame con datos históricos
        
        Returns:
            Net Gain histórico promedio
        """
        df_results = df_results.copy()
        df_results['historical_reward'] = df_results.apply(
            lambda row: self.calculate_observed_reward(
                row.get('outcome_observed', 0),
                row.get('treatment_observed', 0),
                row.get('duration_days', 0)
            ),
            axis=1
        )
        
        historical_gain = df_results['historical_reward'].mean()
        logger.info(f"Net Gain histórico (BAU): ${historical_gain:.2f}")
        
        return float(historical_gain)
    
    def calculate_lift_vs_bau(
        self,
        df_results: pd.DataFrame
    ) -> float:
        """
        Calcula el Lift vs BAU (%).
        
        Args:
            df_results: DataFrame con resultados
        
        Returns:
            Porcentaje de mejora respecto al BAU
        """
        net_gain_model = self.calculate_net_gain(df_results)
        net_gain_historical = self.calculate_historical_net_gain(df_results)
        
        if net_gain_historical == 0:
            logger.warning("Net Gain histórico es 0, no se puede calcular Lift")
            return 0.0
        
        lift = ((net_gain_model - net_gain_historical) / abs(net_gain_historical)) * 100
        
        logger.info(f"Lift vs BAU: {lift:.2f}%")
        
        return float(lift)
    
    def calculate_intervention_percentage(
        self,
        df_results: pd.DataFrame
    ) -> float:
        """
        Calcula el % de Intervenciones (Efficiency).
        
        Args:
            df_results: DataFrame con resultados
        
        Returns:
            Porcentaje de casos donde el modelo recomienda intervenir
        """
        if 'action_model' not in df_results.columns:
            logger.error("action_model no encontrado en el DataFrame")
            return 0.0
        
        intervention_pct = (df_results['action_model'].sum() / len(df_results)) * 100
        
        logger.info(f"% Intervenciones: {intervention_pct:.2f}%")
        
        return float(intervention_pct)
    
    def calculate_violation_percentage(
        self,
        df_results: pd.DataFrame
    ) -> float:
        """
        Calcula el % de Violaciones (Safety/Compliance).
        
        Args:
            df_results: DataFrame con resultados
        
        Returns:
            Porcentaje de acciones que violan reglas de negocio
        """
        if 'action_model' not in df_results.columns:
            logger.error("action_model no encontrado en el DataFrame")
            return 0.0
        
        # Optimización: usar operaciones vectorizadas en lugar de apply
        df_results = df_results.copy()
        
        # Solo verificar seguridad para casos donde action_model == 1
        mask_intervention = df_results['action_model'] == 1
        
        if not mask_intervention.any():
            logger.info("% Violaciones: 0.00% (no hay intervenciones)")
            return 0.0
        
        # Regla 1: No llamar si el estado es "A_Cancelled" o "O_Refused"
        current_state = df_results.get('current_state', '')
        if isinstance(current_state, pd.Series):
            violation_state = df_results['current_state'].astype(str).str.contains(
                'A_Cancelled|O_Refused', case=False, na=False
            )
        else:
            violation_state = pd.Series([False] * len(df_results), index=df_results.index)
        
        # Regla 2: No llamar si ya se llamó en los últimos 2 días
        days_since = df_results.get('days_since_last_intervention', pd.Series([999] * len(df_results)))
        violation_frequency = (days_since < 2) & mask_intervention
        
        # Violación = intervención Y (estado inválido O frecuencia inválida)
        violations = mask_intervention & (violation_state | violation_frequency)
        violation_pct = (violations.sum() / len(df_results)) * 100
        
        logger.info(f"% Violaciones: {violation_pct:.2f}% ({violations.sum()} casos)")
        
        return float(violation_pct)
    
    def calculate_auc_qini(
        self,
        df_results: pd.DataFrame
    ) -> Optional[float]:
        """
        Calcula el AUC-Qini (Area Under Qini Curve).
        
        Solo aplica si el modelo produce un uplift_score continuo.
        
        Args:
            df_results: DataFrame con resultados
        
        Returns:
            AUC-Qini score o None si no hay uplift_score
        """
        if 'uplift_score' not in df_results.columns or df_results['uplift_score'].isna().all():
            logger.warning("uplift_score no encontrado o todos son NaN, AUC-Qini = N/A")
            return None
        
        # Filtrar casos con uplift_score válido
        df_valid = df_results[df_results['uplift_score'].notna()].copy()
        if len(df_valid) == 0:
            logger.warning("No hay casos con uplift_score válido, AUC-Qini = N/A")
            return None
        
        # Ordenar casos por uplift_score descendente
        df_sorted = df_valid.sort_values('uplift_score', ascending=False).copy()
        
        # Optimización: calcular rewards de forma vectorizada
        df_sorted['reward_model'] = df_sorted.apply(
            lambda row: self.calculate_observed_reward(
                row.get('outcome_observed', 0),
                row.get('action_model', 0),
                row.get('duration_days', 0)
            ),
            axis=1
        )
        
        df_sorted['reward_baseline'] = df_sorted.apply(
            lambda row: self.calculate_observed_reward(
                row.get('outcome_observed', 0),
                0,  # No intervenir
                row.get('duration_days', 0)
            ),
            axis=1
        )
        
        # Calcular ganancias acumuladas
        cumulative_gain_model = df_sorted['reward_model'].cumsum().values
        cumulative_gain_baseline = df_sorted['reward_baseline'].cumsum().values
        
        # Calcular área bajo la curva incremental
        # Qini = integral de (gain_model - gain_baseline) sobre k
        incremental_gains = cumulative_gain_model - cumulative_gain_baseline
        n = len(df_sorted)
        auc_qini = np.trapz(incremental_gains) / n if n > 0 else 0.0  # Normalizar por número de casos
        
        logger.info(f"AUC-Qini: {auc_qini:.4f} (calculado sobre {n} casos con uplift_score)")
        
        return float(auc_qini)
    
    def measure_latency(
        self,
        model: Any,
        sample_cases: pd.DataFrame,
        n_iterations: int = 100
    ) -> float:
        """
        Mide la latencia promedio de inferencia.
        
        Args:
            model: Modelo con método predict()
            sample_cases: DataFrame con casos de ejemplo
            n_iterations: Número de iteraciones para promediar
        
        Returns:
            Latencia promedio en milisegundos
        """
        if not hasattr(model, 'predict'):
            logger.warning("Modelo no tiene método predict(), latencia = N/A")
            return None
        
        latencies = []
        
        for _ in range(n_iterations):
            # Seleccionar un caso aleatorio
            sample_case = sample_cases.sample(1)
            
            # Medir tiempo de inferencia
            start = time.time()
            try:
                _ = model.predict(sample_case)
            except Exception as e:
                logger.warning(f"Error en predict: {e}")
                continue
            end = time.time()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        if not latencies:
            logger.warning("No se pudo medir latencia")
            return None
        
        avg_latency = np.mean(latencies)
        logger.info(f"Latencia promedio: {avg_latency:.4f} ms ({n_iterations} iteraciones)")
        
        return float(avg_latency)
    
    def evaluate(
        self,
        df_results: pd.DataFrame,
        model: Optional[Any] = None,
        sample_cases: Optional[pd.DataFrame] = None,
        training_complexity: str = "Media (CPU)"
    ) -> Dict[str, Any]:
        """
        Evalúa todas las métricas y retorna un diccionario con los resultados.
        
        Args:
            df_results: DataFrame con resultados del modelo
            model: Modelo opcional para medir latencia
            sample_cases: Casos de ejemplo para medir latencia
            training_complexity: Complejidad de entrenamiento (string)
        
        Returns:
            Diccionario con las 7 métricas
        """
        logger.info("=" * 80)
        logger.info("EVALUACIÓN DE MÉTRICAS - PRESCRIPTIVE PROCESS MONITORING")
        logger.info("=" * 80)
        
        results = {}
        
        # 1. Net Gain ($)
        try:
            results['net_gain'] = self.calculate_net_gain(df_results)
        except Exception as e:
            logger.error(f"Error calculando Net Gain: {e}")
            results['net_gain'] = None
        
        # 2. Lift vs BAU (%)
        try:
            results['lift_vs_bau'] = self.calculate_lift_vs_bau(df_results)
        except Exception as e:
            logger.error(f"Error calculando Lift vs BAU: {e}")
            results['lift_vs_bau'] = None
        
        # 3. % Intervenciones
        try:
            results['intervention_percentage'] = self.calculate_intervention_percentage(df_results)
        except Exception as e:
            logger.error(f"Error calculando % Intervenciones: {e}")
            results['intervention_percentage'] = None
        
        # 4. % Violaciones
        try:
            results['violation_percentage'] = self.calculate_violation_percentage(df_results)
        except Exception as e:
            logger.error(f"Error calculando % Violaciones: {e}")
            results['violation_percentage'] = None
        
        # 5. AUC-Qini
        try:
            results['auc_qini'] = self.calculate_auc_qini(df_results)
        except Exception as e:
            logger.error(f"Error calculando AUC-Qini: {e}")
            results['auc_qini'] = None
        
        # 6. Latencia (ms)
        if model is not None and sample_cases is not None:
            try:
                results['latency_ms'] = self.measure_latency(model, sample_cases)
            except Exception as e:
                logger.error(f"Error midiendo latencia: {e}")
                results['latency_ms'] = None
        else:
            results['latency_ms'] = None
        
        # 7. Complejidad Entrenamiento
        results['training_complexity'] = training_complexity
        
        logger.info("=" * 80)
        logger.info("RESUMEN DE MÉTRICAS")
        logger.info("=" * 80)
        for key, value in results.items():
            if value is None:
                logger.info(f"  {key}: N/A")
            else:
                logger.info(f"  {key}: {value}")
        
        return results
    
    def evaluate_to_dataframe(
        self,
        df_results: pd.DataFrame,
        model: Optional[Any] = None,
        sample_cases: Optional[pd.DataFrame] = None,
        training_complexity: str = "Media (CPU)"
    ) -> pd.DataFrame:
        """
        Evalúa todas las métricas y retorna un DataFrame con los resultados.
        
        Args:
            df_results: DataFrame con resultados del modelo
            model: Modelo opcional para medir latencia
            sample_cases: Casos de ejemplo para medir latencia
            training_complexity: Complejidad de entrenamiento (string)
        
        Returns:
            DataFrame con una fila y columnas para cada métrica
        """
        results = self.evaluate(df_results, model, sample_cases, training_complexity)
        return pd.DataFrame([results])


def main() -> None:
    """Función principal para ejecutar desde línea de comandos"""
    import sys
    
    # Cargar configuración
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración desde configs/config.yaml")
        sys.exit(1)
    
    # Obtener rutas desde config.yaml
    log_config = config.get("log_config", {})
    bpi2017_config = log_config.get("bpi2017", {})
    
    # Obtener directorio base del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    # Construir rutas absolutas
    xes_path = bpi2017_config.get("xes_path", "logs/BPI2017/BPI Challenge 2017.xes.gz")
    csv_path = bpi2017_config.get("csv_path", "logs/BPI2017/bpi-challenge-2017.csv")
    
    if not os.path.isabs(xes_path):
        xes_path = os.path.join(base_dir, xes_path)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)
    
    logger.info("Benchmark Evaluator para Prescriptive Process Monitoring")
    logger.info("=" * 80)
    logger.info(f"Rutas configuradas:")
    logger.info(f"  XES: {xes_path}")
    logger.info(f"  CSV: {csv_path}")
    
    # Verificar que existe al menos uno de los logs
    if not os.path.exists(xes_path) and not os.path.exists(csv_path):
        logger.error(f"No se encontraron logs BPI 2017:")
        logger.error(f"  XES: {xes_path}")
        logger.error(f"  CSV: {csv_path}")
        logger.error("Por favor, verifica las rutas en configs/config.yaml")
        sys.exit(1)
    
    # Ejemplo de uso
    logger.info("")
    logger.info("Este script define la clase BenchmarkEvaluator.")
    logger.info("Para usarlo, importa la clase y crea un DataFrame con los resultados:")
    logger.info("")
    logger.info("  from benchmark_evaluator import BenchmarkEvaluator")
    logger.info("  evaluator = BenchmarkEvaluator()")
    logger.info("  results = evaluator.evaluate(df_results)")
    logger.info("")
    logger.info("El DataFrame df_results debe tener las siguientes columnas:")
    logger.info("  - case_id: ID del caso")
    logger.info("  - outcome_observed: 1 si préstamo aceptado, 0 si no")
    logger.info("  - treatment_observed: 1 si hubo intervención, 0 si no")
    logger.info("  - duration_days: Duración total del caso")
    logger.info("  - action_model: 1 si modelo recomienda intervenir, 0 si no")
    logger.info("  - propensity_score: Probabilidad histórica de tratamiento (para IPW)")
    logger.info("  - uplift_score: (Opcional) Score de persuasión predicho")
    logger.info("  - current_state: (Opcional) Estado actual del caso")
    logger.info("  - days_since_last_intervention: (Opcional) Días desde última intervención")


if __name__ == "__main__":
    main()
