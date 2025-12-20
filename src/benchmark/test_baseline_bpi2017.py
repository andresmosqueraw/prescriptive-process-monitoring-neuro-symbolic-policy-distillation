#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar el Benchmark Evaluator con el dataset BPI Challenge 2017.
Calcula las métricas del baseline (Business As Usual).
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Agregar src/ al PYTHONPATH para encontrar utils
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)  # src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from benchmark_evaluator import BenchmarkEvaluator
from utils.config import load_config
from utils.logging import setup_logger

# Configurar logger
logger = setup_logger(__name__)


def load_bpi2017_data(csv_path: str) -> pd.DataFrame:
    """
    Carga el dataset BPI 2017 desde CSV.
    
    Args:
        csv_path: Ruta al archivo CSV
    
    Returns:
        DataFrame con los datos cargados
    """
    logger.info(f"Cargando datos desde: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    # Intentar leer el CSV
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        logger.info(f"Columnas disponibles: {list(df.columns)[:10]}...")
        return df
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        raise


def estimate_propensity_score(df_cases: pd.DataFrame) -> np.ndarray:
    """
    Entrena un modelo para estimar P(Treatment | X).
    CORRECCIÓN: No usa 'outcome_observed' para evitar Data Leakage.
    """
    logger.info("Estimando Propensity Scores reales (Sin Data Leakage)...")
    
    feature_cols = []
    
    # Solo usamos características observables ANTES o DURANTE el proceso, no el resultado final
    if 'num_events' in df_cases.columns:
        feature_cols.append('num_events')
    if 'duration_days' in df_cases.columns:
        # Nota: Idealmente deberíamos usar 'elapsed_time_before_treatment', 
        # pero duration_days es un proxy aceptable para el baseline global.
        feature_cols.append('duration_days')
    
    # (Opcional) Si tuvieras 'loan_amount' o 'requested_amount', agrégalos aquí.
    
    if not feature_cols:
        logger.warning("No features, usando promedio.")
        return np.full(len(df_cases), df_cases['treatment_observed'].mean())
    
    features = df_cases[feature_cols].fillna(0)
    target = df_cases['treatment_observed']
    
    try:
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        clf.fit(features, target)
        ps = clf.predict_proba(features)[:, 1]
        return np.clip(ps, 0.05, 0.95) # Clipping conservador
    except Exception as e:
        logger.warning(f"Error estimando PS: {e}")
        return np.full(len(df_cases), target.mean())


def prepare_baseline_dataframe(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Versión CORREGIDA y ESTRICTA para BPI 2017.
    """
    logger.info("Preparando DataFrame (Lógica Estricta BPI 2017)...")
    
    # 1. Identificar columnas (Mapeo robusto)
    column_mapping = {
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    }
    
    # Ajuste dinámico de nombres si difieren en tu CSV
    for k, v in column_mapping.items():
        if v not in df_events.columns:
            # Fallback a columnas comunes si no encuentra las estandar XES
            if k == 'case_id': column_mapping[k] = df_events.columns[0]
            if k == 'activity': column_mapping[k] = 'Activity' if 'Activity' in df_events.columns else df_events.columns[1]
            if k == 'timestamp': column_mapping[k] = 'time:timestamp' # Ajusta según tu CSV real
            
    case_col = column_mapping['case_id']
    act_col = column_mapping['activity']
    time_col = column_mapping['timestamp']
    
    logger.info(f"Usando columnas: Case={case_col}, Activity={act_col}, Time={time_col}")

    # 2. Agrupación Base
    if time_col in df_events.columns:
        # Usar format='mixed' para manejar diferentes formatos de timestamp sin romper
        df_events[time_col] = pd.to_datetime(df_events[time_col], utc=True, format='mixed', errors='coerce')
        
    # Calcular características por caso
    df_cases = df_events.groupby(case_col).agg(
        start_time=(time_col, 'min'),
        end_time=(time_col, 'max'),
        num_events=(act_col, 'count')
    ).reset_index()
    
    # Calcular duración en días
    df_cases['duration_days'] = (df_cases['end_time'] - df_cases['start_time']).dt.total_seconds() / (24 * 3600)
    
    # ---------------------------------------------------------
    # 3. DEFINICIONES ESTRICTAS BPI 2017 (LA CLAVE)
    # ---------------------------------------------------------
    
    # A. Treatment (Intervención Costosa)
    logger.info("Identificando tratamientos (Intervenciones)...")
    
    # Lista EXACTA de actividades que cuestan dinero (Llamadas)
    # No usamos 'contains' genérico para evitar falsos positivos
    treatment_exact_match = [
        'W_Call after offers', 
        'W_Call incomplete files'
    ]
    
    # Crear máscara booleana para eventos de tratamiento
    mask_treatment = df_events[act_col].isin(treatment_exact_match)
    
    # Debug: ¿Cuántos eventos son tratamiento?
    n_treat_events = mask_treatment.sum()
    logger.info(f"Eventos de tratamiento encontrados (Exact Match): {n_treat_events}")
    
    if n_treat_events == 0:
        logger.warning("⚠️ No se encontraron eventos exactos 'W_Call...'. Intentando búsqueda laxa (W_Call)...")
        # Fallback laxo solo si el estricto falla totalmente
        mask_treatment = df_events[act_col].astype(str).str.contains('W_Call', case=False, na=False)
        # Mostrar qué encontró el fallback para depurar
        captured_activities = df_events.loc[mask_treatment, act_col].unique()
        logger.info(f"Actividades capturadas por fallback: {captured_activities}")
        
    # Identificar IDs de casos tratados
    treated_case_ids = df_events.loc[mask_treatment, case_col].unique()
    
    # Asignar 1 o 0
    df_cases['treatment_observed'] = df_cases[case_col].isin(treated_case_ids).astype(int)
    
    # Validación de Sanidad
    pct_treated = df_cases['treatment_observed'].mean() * 100
    logger.info(f"Porcentaje de casos tratados: {pct_treated:.2f}%")
    
    if pct_treated > 90:
        # Nota: En BPI 2017, >90% de casos tratados es ESPERADO
        # Casi todos los casos requieren alguna intervención manual en este proceso
        logger.info(f"ℹ️  Nota: {pct_treated:.2f}% de casos tratados es ESPERADO en BPI 2017.")
        logger.info("   Casi todos los casos requieren alguna intervención manual en este proceso.")
        # Solo warning si es >99.5% (posible error en nombres de actividades)
        if pct_treated > 99.5:
            logger.warning("🚨 ALERTA: % Tratados > 99.5%. Revisar nombres de actividades.")
            captured_activities = df_events.loc[mask_treatment, act_col].unique()
            logger.warning(f"Actividades capturadas como tratamiento: {captured_activities}")

    # B. Outcome (Éxito del Negocio)
    # El éxito es O_Accepted
    success_activities = ['O_Accepted']
    
    success_case_ids = df_events[
        df_events[act_col].isin(success_activities)
    ][case_col].unique()
    
    df_cases['outcome_observed'] = df_cases[case_col].isin(success_case_ids).astype(int)
    
    # ---------------------------------------------------------
    
    # 4. Propensity Score (Usando la función corregida anteriormente)
    try:
        df_cases['propensity_score'] = estimate_propensity_score(df_cases)
    except Exception as e:
        logger.warning(f"Error en propensity score: {e}. Usando promedio.")
        df_cases['propensity_score'] = df_cases['treatment_observed'].mean()

    # 5. Configuración Baseline
    # El modelo "Baseline" simplemente hace lo que se hizo históricamente
    df_cases['action_model'] = df_cases['treatment_observed']
    
    # 6. Safety Check Data (Estado actual y última intervención)
    # Para el baseline histórico, esto es solo informativo
    # Nota: En un modelo real, 'current_state' debería ser calculado dinámicamente
    # basado en la última actividad del caso (ej: 'A_Cancelled', 'O_Refused', 'O_Accepted', etc.)
    df_cases['current_state'] = 'Closed'  # Simplificación para baseline
    df_cases['days_since_last_intervention'] = 999  # Valor seguro para baseline 
    
    # 7. Uplift score: None para baseline
    df_cases['uplift_score'] = None
    
    # 8. Renombrar case_id para consistencia
    df_cases = df_cases.rename(columns={case_col: 'case_id'})
    
    # 9. Seleccionar columnas necesarias para el evaluador
    result_columns = [
        'case_id',
        'outcome_observed',
        'treatment_observed',
        'duration_days',
        'action_model',
        'propensity_score',
        'uplift_score',
        'current_state',
        'days_since_last_intervention'
    ]
    
    df_result = df_cases[result_columns].copy()
    
    logger.info(f"DataFrame preparado: {len(df_result)} casos")
    logger.info(f"Estadísticas:")
    logger.info(f"  Outcome promedio: {df_result['outcome_observed'].mean():.2%}")
    logger.info(f"  Treatment promedio: {df_result['treatment_observed'].mean():.2%}")
    logger.info(f"  Duración promedio: {df_result['duration_days'].mean():.2f} días")
    
    return df_result


def process_baseline_for_log(csv_path: str, log_name: str, project_root: str) -> None:
    """
    Procesa el baseline para un log específico y guarda los resultados.
    
    Args:
        csv_path: Ruta absoluta al archivo CSV del log
        log_name: Nombre del log (sin extensión) para crear la carpeta de salida
        project_root: Directorio raíz del proyecto
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PROCESANDO BASELINE: {log_name}")
    logger.info("=" * 80)
    logger.info(f"Archivo: {csv_path}")
    
    # Preparar DataFrame
    try:
        df_events = load_bpi2017_data(csv_path)
        df_results = prepare_baseline_dataframe(df_events)
    except Exception as e:
        logger.error(f"Error procesando {log_name}: {e}")
        return
    
    # Crear evaluador (puede usar configuración personalizada si está disponible)
    # Por ahora usamos valores por defecto, pero se puede extender para leer de benchmark_config
    evaluator = BenchmarkEvaluator()
    
    # 1. Calcular métricas estándar (IPW, etc)
    results = evaluator.evaluate(
        df_results=df_results,
        model=None,  # No hay modelo para baseline
        sample_cases=None,
        training_complexity="N/A (Baseline)"
    )
    
    # 2. Calcular la VERDAD HISTÓRICA (Ground Truth)
    # Para el baseline, no queremos una estimación IPW, queremos el dato real.
    historical_gain = evaluator.calculate_historical_net_gain(df_results)
    
    # 3. SOBREESCRIBIR para el reporte oficial del Baseline
    results['net_gain'] = historical_gain
    results['lift_vs_bau'] = 0.0  # Por definición, el baseline tiene 0% lift contra sí mismo
    
    # Mostrar resultados formateados
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"RESULTADOS BASELINE OFICIALES (GROUND TRUTH) - {log_name}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("💰 Net Gain ($):        ${:.2f}".format(results['net_gain'])) 
    logger.info("📈 Lift vs BAU (%):     {:.2f}%".format(results['lift_vs_bau']))
    logger.info("📉 % Intervenciones:    {:.2f}%".format(results.get('intervention_percentage', 0)))
    logger.info("🛡️  % Violaciones:       {:.2f}%".format(results.get('violation_percentage', 0)))
    logger.info("🐢 Latencia (ms):       0.0000")  # Baseline no tiene latencia de inferencia
    logger.info("")
    
    # Guardar resultados en CSV
    # Aseguramos que la latencia sea 0 explícitamente para el CSV
    results['latency_ms'] = 0.0
    
    # Crear DF manual para asegurar que se guardan los valores corregidos
    results_df = pd.DataFrame([results])
    
    # Guardar en carpeta específica del log
    # Intentar leer configuración de salida del benchmark, sino usar por defecto
    benchmark_config = load_benchmark_config()
    if benchmark_config:
        output_config = benchmark_config.get("output", {})
        base_dir = output_config.get("base_dir", "results/benchmark")
        metrics_filename = output_config.get("metrics_filename", "baseline_metrics.csv")
    else:
        base_dir = "results/benchmark"
        metrics_filename = "baseline_metrics.csv"
    
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(project_root, base_dir)
    
    output_dir = os.path.join(base_dir, log_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, metrics_filename)
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ Resultados guardados en: {output_path}")


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
        logger.warning(f"No se encontró {config_path}, usando configuración por defecto")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error cargando configuración del benchmark: {e}")
        return None


def main() -> None:
    """Función principal - Procesa ambos logs (completo y sample)"""
    logger.info("=" * 80)
    logger.info("TEST BASELINE - BPI CHALLENGE 2017")
    logger.info("Procesando log completo y sample")
    logger.info("=" * 80)
    
    # Encontrar el directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = src/benchmark/
    src_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    # Cargar configuración del benchmark
    benchmark_config = load_benchmark_config()
    
    # Si no hay config del benchmark, intentar cargar la config principal como fallback
    if benchmark_config is None:
        logger.info("Intentando cargar configuración principal como fallback...")
        config = load_config()
        if config is None:
            logger.error("No se pudo cargar ninguna configuración")
            sys.exit(1)
        # Usar la lógica anterior como fallback
        log_config = config.get("log_config", {})
        bpi2017_config = log_config.get("bpi2017", {})
        
        logs_to_process = []
        csv_path_full = bpi2017_config.get("csv_path")
        if csv_path_full:
            if not os.path.isabs(csv_path_full):
                csv_path_full = os.path.join(project_root, csv_path_full)
            logs_to_process.append(csv_path_full)
        
        log_path = log_config.get("log_path")
        if log_path:
            if not os.path.isabs(log_path):
                log_path = os.path.join(project_root, log_path)
            if log_path != csv_path_full and log_path.endswith('.csv'):
                logs_to_process.append(log_path)
    else:
        # Usar configuración del benchmark
        logs_config = benchmark_config.get("logs", {})
        processing_config = benchmark_config.get("processing", {})
        process_both = processing_config.get("process_both_logs", True)
        
        logs_to_process = []
        
        # Log completo
        if process_both or not logs_config.get("sample_log"):
            full_log_config = logs_config.get("full_log", {})
            csv_path_full = full_log_config.get("csv_path")
            if csv_path_full:
                if not os.path.isabs(csv_path_full):
                    csv_path_full = os.path.join(project_root, csv_path_full)
                logs_to_process.append(csv_path_full)
                logger.info(f"📋 Log completo: {csv_path_full}")
        
        # Log sample
        if process_both:
            sample_log_config = logs_config.get("sample_log", {})
            csv_path_sample = sample_log_config.get("csv_path")
            if csv_path_sample:
                if not os.path.isabs(csv_path_sample):
                    csv_path_sample = os.path.join(project_root, csv_path_sample)
                if csv_path_sample not in logs_to_process:
                    logs_to_process.append(csv_path_sample)
                    logger.info(f"📋 Log sample: {csv_path_sample}")
    
    # Si no se encontraron logs, usar valores por defecto
    if not logs_to_process:
        logger.warning("⚠️  No se encontraron rutas en la configuración, usando valores por defecto")
        default_full = os.path.join(project_root, "logs", "BPI2017", "bpi-challenge-2017.csv")
        default_sample = os.path.join(project_root, "logs", "BPI2017", "bpi-challenge-2017-sample.csv")
        if os.path.exists(default_full):
            logs_to_process.append(default_full)
        if os.path.exists(default_sample):
            logs_to_process.append(default_sample)
    
    # Procesar cada log
    processed_count = 0
    for csv_path in logs_to_process:
        if os.path.exists(csv_path):
            log_name = os.path.splitext(os.path.basename(csv_path))[0]
            # Remover extensión adicional si es .gz
            if log_name.endswith('.xes') or log_name.endswith('.csv'):
                log_name = os.path.splitext(log_name)[0]
            process_baseline_for_log(csv_path, log_name, project_root)
            processed_count += 1
        else:
            logger.warning(f"⚠️  No se encontró el log: {csv_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ PROCESAMIENTO COMPLETADO ({processed_count} log(s) procesado(s))")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
