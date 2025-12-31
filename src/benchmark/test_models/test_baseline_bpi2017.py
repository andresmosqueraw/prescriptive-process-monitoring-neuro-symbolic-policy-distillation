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
import argparse
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
from utils.logger_utils import setup_logger

# Configurar logger
logger = setup_logger(__name__)

def load_benchmark_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Carga la configuración del benchmark desde benchmark_config.yaml.
    """
    if config_path is None:
        # Determinar directorio base del proyecto
        script_dir = os.path.dirname(os.path.abspath(__file__))
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

def load_bpi2017_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"Cargando datos desde: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        raise

def estimate_propensity_score(df_cases: pd.DataFrame) -> np.ndarray:
    logger.info("Estimando Propensity Scores reales (Sin Data Leakage)...")
    feature_cols = []
    if 'num_events' in df_cases.columns: feature_cols.append('num_events')
    if 'duration_days' in df_cases.columns: feature_cols.append('duration_days')
    
    if not feature_cols:
        logger.warning("No features, usando promedio.")
        return np.full(len(df_cases), df_cases['treatment_observed'].mean())
    
    features = df_cases[feature_cols].fillna(0)
    target = df_cases['treatment_observed']

    # Check si hay al menos 2 clases
    if len(np.unique(target)) < 2:
        logger.warning(f"Solo hay una clase en target ({np.unique(target)}). Usando probabilidad promedio constante.")
        # Si todos son 1, PS = 0.99. Si todos son 0, PS = 0.01.
        mean_val = np.mean(target)
        ps_val = 0.99 if mean_val > 0.5 else 0.01
        return np.full(len(df_cases), ps_val)
    
    try:
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        clf.fit(features, target)
        ps = clf.predict_proba(features)[:, 1]
        return np.clip(ps, 0.05, 0.95)
    except Exception as e:
        logger.warning(f"Error estimando PS: {e}")
        return np.full(len(df_cases), target.mean())

def prepare_baseline_dataframe(df_events: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparando DataFrame (Lógica Estricta BPI 2017)...")
    
    # Mapeo robusto
    column_mapping = {
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    }
    for k, v in column_mapping.items():
        if v not in df_events.columns:
            if k == 'case_id': column_mapping[k] = df_events.columns[0]
            if k == 'activity': column_mapping[k] = 'Activity' if 'Activity' in df_events.columns else df_events.columns[1]
            if k == 'timestamp': column_mapping[k] = 'time:timestamp'
            
    case_col = column_mapping['case_id']
    act_col = column_mapping['activity']
    time_col = column_mapping['timestamp']
    
    # Agrupación
    if time_col in df_events.columns:
        df_events[time_col] = pd.to_datetime(df_events[time_col], utc=True, format='mixed', errors='coerce')
        
    df_cases = df_events.groupby(case_col).agg(
        start_time=(time_col, 'min'),
        end_time=(time_col, 'max'),
        num_events=(act_col, 'count')
    ).reset_index()
    
    df_cases['duration_days'] = (df_cases['end_time'] - df_cases['start_time']).dt.total_seconds() / (24 * 3600)
    
    # Tratamiento (Exact Match)
    treatment_exact_match = ['W_Call after offers', 'W_Call incomplete files']
    mask_treatment = df_events[act_col].isin(treatment_exact_match)
    
    if mask_treatment.sum() == 0:
        mask_treatment = df_events[act_col].astype(str).str.contains('W_Call', case=False, na=False)
        
    treated_case_ids = df_events.loc[mask_treatment, case_col].unique()
    df_cases['treatment_observed'] = df_cases[case_col].isin(treated_case_ids).astype(int)
    
    # Outcome
    success_activities = ['O_Accepted']
    success_case_ids = df_events[df_events[act_col].isin(success_activities)][case_col].unique()
    df_cases['outcome_observed'] = df_cases[case_col].isin(success_case_ids).astype(int)
    
    # Propensity Score & Config Baseline
    try:
        df_cases['propensity_score'] = estimate_propensity_score(df_cases)
    except Exception:
        df_cases['propensity_score'] = df_cases['treatment_observed'].mean()

    df_cases['action_model'] = df_cases['treatment_observed']
    df_cases['current_state'] = 'Closed'
    df_cases['days_since_last_intervention'] = 999
    df_cases['uplift_score'] = None
    
    df_cases = df_cases.rename(columns={case_col: 'case_id'})
    
    return df_cases[['case_id', 'outcome_observed', 'treatment_observed', 'duration_days', 
                     'action_model', 'propensity_score', 'uplift_score', 'current_state', 
                     'days_since_last_intervention']]

def process_baseline_for_log(csv_path: str, log_name: str, project_root: str) -> None:
    logger.info(f"\n{'='*80}\nPROCESANDO BASELINE: {log_name}\n{'='*80}")
    
    try:
        df_events = load_bpi2017_data(csv_path)
        df_results = prepare_baseline_dataframe(df_events)
    except Exception as e:
        logger.error(f"Error procesando {log_name}: {e}")
        return
    
    evaluator = BenchmarkEvaluator()
    
    # 1. Métricas estándar
    results = evaluator.evaluate(df_results, training_complexity="N/A (Baseline)")
    
    # 2. Verdad Histórica (Ground Truth)
    historical_gain = evaluator.calculate_historical_net_gain(df_results)
    results['net_gain'] = historical_gain
    results['lift_vs_bau'] = 0.0
    results['latency_ms'] = 0.0
    
    logger.info(f"\nRESULTADOS OFICIALES ({log_name}):")
    logger.info(f"💰 Net Gain: ${results['net_gain']:.2f}")
    logger.info(f"📉 Intervenciones: {results.get('intervention_percentage', 0):.2f}%")
    
    # Guardar
    benchmark_config = load_benchmark_config()
    base_dir = benchmark_config.get("output", {}).get("base_dir", "results/benchmark") if benchmark_config else "results/benchmark"
    metrics_filename = benchmark_config.get("output", {}).get("metrics_filename", "baseline_metrics.csv") if benchmark_config else "baseline_metrics.csv"
    
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(project_root, base_dir)
    
    output_dir = os.path.join(base_dir, log_name)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([results]).to_csv(os.path.join(output_dir, metrics_filename), index=False)
    logger.info(f"✅ Guardado en: {os.path.join(output_dir, metrics_filename)}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Test Baseline para BPI Challenge 2017')
    parser.add_argument('--test', action='store_true', 
                       help='Procesar específicamente el archivo de test (bpi2017_test.csv)')
    parser.add_argument('--train', action='store_true',
                       help='Procesar específicamente el archivo de train (bpi2017_train.csv)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Ruta a un archivo CSV específico a procesar')
    args = parser.parse_args()
    
    logger.info(f"{'='*80}\nTEST BASELINE - BPI CHALLENGE 2017\n{'='*80}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark/test_models/
    benchmark_dir = os.path.dirname(script_dir)  # src/benchmark/
    src_dir = os.path.dirname(benchmark_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    logs_to_process = []
    
    # Si se especifica --test, procesar solo el archivo de test
    if args.test:
        test_path = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_test.csv")
        if os.path.exists(test_path):
            logs_to_process.append(test_path)
            logger.info(f"🎯 Modo TEST: Procesando archivo de test")
        else:
            logger.error(f"❌ No se encontró el archivo de test: {test_path}")
            return
    
    # Si se especifica --train, procesar solo el archivo de train
    elif args.train:
        train_path = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_train.csv")
        if os.path.exists(train_path):
            logs_to_process.append(train_path)
            logger.info(f"🎯 Modo TRAIN: Procesando archivo de train")
        else:
            logger.error(f"❌ No se encontró el archivo de train: {train_path}")
            return
    
    # Si se especifica --csv, procesar ese archivo
    elif args.csv:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(project_root, csv_path)
        if os.path.exists(csv_path):
            logs_to_process.append(csv_path)
            logger.info(f"🎯 Procesando archivo CSV especificado: {csv_path}")
        else:
            logger.error(f"❌ No se encontró el archivo: {csv_path}")
            return
    
    # Si no se especifica nada, usar la lógica original (config o fallback)
    else:
        benchmark_config = load_benchmark_config()
        
        if benchmark_config:
            logs_conf = benchmark_config.get("logs", {})
            if benchmark_config.get("processing", {}).get("process_both_logs", True):
                if logs_conf.get("full_log", {}).get("csv_path"): 
                    path = logs_conf["full_log"]["csv_path"]
                    if not os.path.isabs(path): path = os.path.join(project_root, path)
                    logs_to_process.append(path)
                if logs_conf.get("sample_log", {}).get("csv_path"):
                    path = logs_conf["sample_log"]["csv_path"]
                    if not os.path.isabs(path): path = os.path.join(project_root, path)
                    logs_to_process.append(path)
        
        if not logs_to_process:
            # Fallback: buscar archivo de test por defecto
            test_path = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_test.csv")
            if os.path.exists(test_path):
                logs_to_process.append(test_path)
                logger.info(f"📋 Usando archivo de test por defecto: {test_path}")
            else:
                # Último fallback: archivo completo
                default_full = os.path.join(project_root, "logs/BPI2017/bpi-challenge-2017.csv")
                if os.path.exists(default_full): 
                    logs_to_process.append(default_full)
                    logger.info(f"📋 Usando archivo completo por defecto: {default_full}")
        
    # Procesar los archivos encontrados
    for csv_path in logs_to_process:
        if os.path.exists(csv_path):
            log_name = os.path.splitext(os.path.basename(csv_path))[0].replace(".csv", "").replace(".xes", "")
            process_baseline_for_log(csv_path, log_name, project_root)
        else:
            logger.warning(f"Log no encontrado: {csv_path}")

if __name__ == "__main__":
    main()