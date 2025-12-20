#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUACIÓN FINAL: CAUSAL-GYM vs BASELINE
----------------------------------------
Este script carga el modelo destilado (Decision Tree) y lo evalúa sobre el log histórico.
Reconstruye los estados del log, genera predicciones y calcula el Net Gain usando IPW.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import yaml

# Agregar directorios al path
script_dir = os.path.dirname(os.path.abspath(__file__)) # src/benchmark
src_dir = os.path.dirname(script_dir)                   # src/
project_root = os.path.dirname(src_dir)                 # root
if src_dir not in sys.path: sys.path.insert(0, src_dir)

from utils.config import load_config
from utils.logging import setup_logger
from benchmark_evaluator import BenchmarkEvaluator
from test_baseline_bpi2017 import load_bpi2017_data, estimate_propensity_score, prepare_baseline_dataframe

logger = setup_logger(__name__)

def load_trained_model(model_path):
    """Carga el modelo .pkl generado en Fase 3"""
    if not os.path.exists(model_path):
        logger.error(f"❌ No se encontró el modelo en: {model_path}")
        logger.error("Ejecuta primero: python src/causal-gym/distill_policy.py")
        sys.exit(1)
    
    logger.info(f"🧠 Cargando modelo Causal-Gym: {os.path.basename(model_path)}")
    return joblib.load(model_path)

def reconstruct_state_features(df_events, case_col, act_col):
    """
    Reconstruye el string de estado 'loc=...|amt=...|type=...' desde el CSV crudo.
    Debe coincidir EXACTAMENTE con la lógica de train_agent_in_gym.py.
    """
    logger.info("🔧 Reconstruyendo estados para el modelo...")
    
    # Check if required columns exist, add default if missing (for robustness)
    if 'case:RequestedAmount' not in df_events.columns:
        logger.warning("'case:RequestedAmount' no encontrado. Usando 0.0")
        df_events['case:RequestedAmount'] = 0.0
    if 'case:ApplicationType' not in df_events.columns:
        logger.warning("'case:ApplicationType' no encontrado. Usando 'Unknown'")
        df_events['case:ApplicationType'] = 'Unknown'

    # Pre-calcular atributos estáticos por caso (Monto y Tipo)
    # Asumimos que case:RequestedAmount y case:ApplicationType son constantes por caso
    case_attrs = df_events.groupby(case_col).first()[['case:RequestedAmount', 'case:ApplicationType']].reset_index()
    
    # Mapeo rápido para vectorización
    case_amount_map = dict(zip(case_attrs[case_col], case_attrs['case:RequestedAmount']))
    case_type_map = dict(zip(case_attrs[case_col], case_attrs['case:ApplicationType']))
    
    def _build_state(row):
        c_id = row[case_col]
        loc = row[act_col]
        # Manejo robusto de nulos
        amt = case_amount_map.get(c_id, 0)
        tpe = case_type_map.get(c_id, 'Unknown')
        
        # Formato exacto usado en training: f"loc={id}|amt={amt}|type={type}"
        return f"loc={loc}|amt={amt}|type={tpe}"

    # Aplicar a todo el dataframe (puede tardar un poco en logs gigantes)
    # Optimización: Solo necesitamos predecir en los puntos de decisión relevantes
    # Pero para simplificar, generamos para todo y luego filtramos
    return df_events.apply(_build_state, axis=1)

def apply_model_policy(df_events, model, case_col, act_col):
    """
    Usa el modelo para predecir acciones (Intervenir o No) sobre el log histórico.
    """
    # 1. Generar Features de Estado
    state_features = reconstruct_state_features(df_events, case_col, act_col)
    
    # 2. Predecir
    logger.info("🔮 Generando predicciones del modelo...")
    predicted_actions_str = model.predict(state_features)
    
    # --- FIX CRÍTICO ---
    # Identificar cuál es la acción de intervención.
    # El modelo predice clases. Veamos cuáles son.
    unique_actions = np.unique(predicted_actions_str)
    logger.info(f"DEBUG: Acciones del modelo: {unique_actions}")
    
    # Si hay acciones que parecen IDs de nodos (empiezan con node_), 
    # y sabemos que el baseline hacía 100% intervenciones,
    # asumimos que la acción predominante o la que no es 'None' es la intervención.
    
    # Estrategia: Si el string NO es "None" o vacío, es una intervención.
    # (En Prosimos, no hacer nada suele ser seguir el flujo default o un nodo distinto)
    
    binary_actions = []
    for a in predicted_actions_str:
        a_str = str(a)
        # Criterio ampliado: "Call" O es un ID de nodo activo
        if "Call" in a_str or "node_" in a_str:
            binary_actions.append(1)
        else:
            binary_actions.append(0)
            
    binary_actions = np.array(binary_actions)
    
    # Agregar predicción al DF original temporalmente
    df_events['model_action_binary'] = binary_actions
    
    # 4. Agrupar por Caso (Políticas a nivel de caso)
    # Si el modelo recomienda llamar AL MENOS UNA VEZ en el caso, action_model = 1
    case_decisions = df_events.groupby(case_col)['model_action_binary'].max().reset_index()
    case_decisions.rename(columns={'model_action_binary': 'action_model'}, inplace=True)
    
    return case_decisions

def main():
    logger.info(f"{'='*80}\nEVALUACIÓN FINAL: CAUSAL-GYM vs BASELINE\n{'='*80}")
    
    # 1. Configuración y Rutas
    config = load_config()
    # Usar log completo por defecto, o sample si se prefiere
    # log_path = "logs/BPI2017/bpi-challenge-2017.csv" 
    log_path = config["log_config"]["log_path"] # Usa el que esté en config.yaml
    if not os.path.isabs(log_path): log_path = os.path.join(project_root, log_path)
    
    # Ruta del modelo
    model_path = os.path.join(project_root, "results/distill/bpi-challenge-2017-sample/distill/final_policy_model.pkl")
    # Ajuste para buscar el modelo si la ruta exacta varía
    if not os.path.exists(model_path):
        # Intentar búsqueda genérica
        for root, dirs, files in os.walk(os.path.join(project_root, "results")):
            if "final_policy_model.pkl" in files:
                model_path = os.path.join(root, "final_policy_model.pkl")
                break
    
    # 2. Cargar Datos y Modelo
    df_events = load_bpi2017_data(log_path)
    model = load_trained_model(model_path)
    
    # 3. Preparar DataFrame Base (Ground Truth & Propensity)
    # Reutilizamos la lógica robusta del baseline para obtener outcomes y tratamientos reales
    df_results = prepare_baseline_dataframe(df_events)
    
    # 4. Inyectar Decisiones del Modelo (Sobreescribir action_model del baseline)
    # Mapeo de columnas para reconstrucción de estado
    case_col = 'case:concept:name' if 'case:concept:name' in df_events.columns else df_events.columns[0]
    act_col = 'concept:name' if 'concept:name' in df_events.columns else df_events.columns[1]
    
    model_decisions = apply_model_policy(df_events, model, case_col, act_col)
    
    # Merge decisiones del modelo en df_results
    # df_results tiene ['case_id', 'outcome_observed', ...]
    # model_decisions tiene ['case_id', 'action_model']
    
    # Eliminar el action_model del baseline (que era copy-paste del histórico)
    if 'action_model' in df_results.columns:
        del df_results['action_model']
        
    df_final = df_results.merge(
        model_decisions.rename(columns={case_col: 'case_id'}), 
        on='case_id', 
        how='left'
    )
    df_final['action_model'] = df_final['action_model'].fillna(0).astype(int)
    
    # 5. Calcular Métricas con BenchmarkEvaluator
    logger.info("📊 Calculando métricas de rendimiento...")
    evaluator = BenchmarkEvaluator()
    metrics = evaluator.evaluate(df_final, training_complexity="Baja (CPU - Tree)")
    
    # 6. Comparar con Baseline (Cargar valor histórico)
    # Leemos el archivo CSV generado anteriormente para tener el dato exacto
    baseline_csv = os.path.join(project_root, "results/benchmark/bpi-challenge-2017-sample/baseline_metrics.csv")
    if not os.path.exists(baseline_csv):
        baseline_csv = os.path.join(project_root, "results/benchmark/baseline_metrics.csv")
        
    baseline_gain = 0.0
    if os.path.exists(baseline_csv):
        try:
            base_df = pd.read_csv(baseline_csv)
            baseline_gain = base_df['net_gain'].iloc[0]
            logger.info(f"📉 Baseline Net Gain cargado: ${baseline_gain:.2f}")
        except:
            logger.warning("No se pudo leer baseline_metrics.csv, asumiendo 0 para Lift")
    else:
        logger.warning("Archivo baseline_metrics.csv no encontrado.")

    # Recalcular Lift real usando el Net Gain del modelo vs Baseline Histórico
    model_gain = metrics['net_gain']
    if baseline_gain != 0:
        real_lift = ((model_gain - baseline_gain) / abs(baseline_gain)) * 100
        metrics['lift_vs_bau'] = real_lift
    
    # 7. Reporte Final
    print("\n" + "="*60)
    print("🏆 RESULTADOS FINALES: CAUSAL-GYM")
    print("="*60)
    print(f"💰 Net Gain (Modelo):     ${metrics['net_gain']:.2f}")
    print(f"📉 Baseline Gain:         ${baseline_gain:.2f}")
    print(f"🚀 LIFT REAL:             {metrics['lift_vs_bau']:+.2f}%")
    print(f"🎯 % Intervenciones:      {metrics['intervention_percentage']:.2f}%")
    print(f"🛡️  % Violaciones:         {metrics['violation_percentage']:.2f}%")
    print("="*60 + "\n")
    
    # Guardar
    out_dir = os.path.join(project_root, "results/benchmark/causal_gym")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "causal_gym_metrics.csv"), index=False)
    logger.info(f"Resultados guardados en: {out_dir}")

if __name__ == "__main__":
    main()