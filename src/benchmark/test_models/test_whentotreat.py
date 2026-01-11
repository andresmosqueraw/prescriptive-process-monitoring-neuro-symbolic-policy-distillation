#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUACIÓN FINAL: WhenToTreat vs BASELINE
----------------------------------------
Este script entrena/evalúa el modelo WhenToTreat (CausalForest) sobre el log histórico.
Genera predicciones basadas en Treatment Effects y calcula el Net Gain usando IPW.
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import pickle
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Dict, Any

# Agregar directorios al path (IMPORTANTE: hacer esto PRIMERO para evitar conflictos)
script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark/test_models/
benchmark_dir = os.path.dirname(script_dir)  # src/benchmark/
src_dir = os.path.dirname(benchmark_dir)  # src/
project_root = os.path.dirname(src_dir)  # root
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar utils del proyecto ACTUAL primero (antes de agregar WhenToTreat al path)
from utils.config import load_config
from utils.logger_utils import setup_logger

# Agregar benchmark al path para importar benchmark_evaluator
benchmark_dir = os.path.dirname(script_dir)  # src/benchmark/
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

from benchmark_evaluator import BenchmarkEvaluator
from test_baseline_bpi2017 import load_bpi2017_data, estimate_propensity_score, prepare_baseline_dataframe

# AHORA intentar importar WhenToTreat (después de importar utils del proyecto actual)
wtt_available = False
wtt_path = "/home/andrew/Documents/asistencia-graduada-phd-oscar/paper1/prescriptive-process-monitoring-models/WhenToTreat"
if os.path.exists(wtt_path):
    try:
        # Agregar al path pero al final para evitar conflictos con utils
        if wtt_path not in sys.path:
            sys.path.append(wtt_path)  # append en lugar de insert para menor prioridad
        from causal_estimators.forest_estimators import CausalForest
        from causal_estimators.wtt_estimator import wtt_estimator
        wtt_available = True
        logger = setup_logger(__name__)
        logger.info("✅ WhenToTreat disponible")
    except ImportError as e:
        # Si falla, usar econml directamente
        try:
            import econml.grf
            wtt_available = True
            logger = setup_logger(__name__)
            logger.info(f"⚠️  WhenToTreat no disponible: {e}")
            logger.info("   Usando econml directamente")
        except ImportError:
            logger = setup_logger(__name__)
            logger.error("❌ econml no está instalado. Instala con: pip install econml")
            sys.exit(1)
else:
    # Si WhenToTreat no está disponible, usar econml directamente
    try:
        import econml.grf
        wtt_available = True
        logger = setup_logger(__name__)
        logger.info("⚠️  Repositorio WhenToTreat no encontrado")
        logger.info("   Usando econml directamente")
    except ImportError:
        logger = setup_logger(__name__)
        logger.error("❌ econml no está instalado. Instala con: pip install econml")
        sys.exit(1)

# Configurar logger si no se configuró antes
if 'logger' not in locals():
    logger = setup_logger(__name__)

logger = setup_logger(__name__)


def prepare_wtt_features(df_events: pd.DataFrame) -> tuple:
    """
    Prepara features básicas para WhenToTreat desde el log de eventos.
    Versión simplificada que extrae features básicas del CSV.
    
    Returns:
        Tupla (df_features, case_col) con features preparadas y nombre de columna de caso
    """
    logger.info("🔧 Preparando features para WhenToTreat...")
    
    # Identificar columnas
    case_col = 'case:concept:name' if 'case:concept:name' in df_events.columns else df_events.columns[0]
    act_col = 'concept:name' if 'concept:name' in df_events.columns else df_events.columns[1]
    time_col = 'time:timestamp' if 'time:timestamp' in df_events.columns else None
    resource_col = 'org:resource' if 'org:resource' in df_events.columns else None
    
    # Convertir timestamp si existe
    if time_col and time_col in df_events.columns:
        df_events[time_col] = pd.to_datetime(df_events[time_col], utc=True, format='mixed', errors='coerce')
    
    # Agrupar por caso para obtener features agregadas
    agg_dict = {
        act_col: 'count'  # num_events
    }
    
    if resource_col and resource_col in df_events.columns:
        agg_dict[resource_col] = 'nunique'  # num_unique_resources
    
    df_cases = df_events.groupby(case_col).agg(agg_dict).reset_index()
    df_cases.columns = [case_col] + [c for c in df_cases.columns if c != case_col]
    
    # Renombrar columnas
    if act_col in df_cases.columns:
        df_cases = df_cases.rename(columns={act_col: 'num_events'})
    if resource_col and resource_col in df_cases.columns:
        df_cases = df_cases.rename(columns={resource_col: 'num_unique_resources'})
    
    # Features estáticas por caso (solo numéricas)
    static_numeric = {
        'case:RequestedAmount': 'case:RequestedAmount',
        'RequestedAmount': 'case:RequestedAmount',  # Alternativa sin prefijo
    }
    
    static_categorical = {
        'case:ApplicationType': 'case:ApplicationType',
        'ApplicationType': 'case:ApplicationType',
        'case:LoanGoal': 'case:LoanGoal',
        'LoanGoal': 'case:LoanGoal'
    }
    
    # Agregar features numéricas estáticas
    for orig_col, new_col in static_numeric.items():
        if orig_col in df_events.columns:
            case_feat = df_events.groupby(case_col)[orig_col].first().reset_index()
            case_feat = case_feat.rename(columns={orig_col: new_col})
            df_cases = df_cases.merge(case_feat, on=case_col, how='left')
            break  # Solo tomar la primera que encuentre
    
    # Agregar features categóricas estáticas (para one-hot encoding)
    for orig_col, new_col in static_categorical.items():
        if orig_col in df_events.columns:
            case_feat = df_events.groupby(case_col)[orig_col].first().reset_index()
            case_feat = case_feat.rename(columns={orig_col: new_col})
            df_cases = df_cases.merge(case_feat, on=case_col, how='left')
            break  # Solo tomar la primera que encuentre
    
    # Calcular duración si hay timestamp
    if time_col and time_col in df_events.columns:
        case_times = df_events.groupby(case_col)[time_col].agg(['min', 'max']).reset_index()
        case_times['duration_days'] = (case_times['max'] - case_times['min']).dt.total_seconds() / (24 * 3600)
        df_cases = df_cases.merge(case_times[[case_col, 'duration_days']], on=case_col, how='left')
    else:
        df_cases['duration_days'] = 0.0
    
    # Features numéricas básicas
    numeric_features = []
    
    # RequestedAmount
    if 'case:RequestedAmount' in df_cases.columns:
        numeric_features.append('case:RequestedAmount')
        df_cases['case:RequestedAmount'] = pd.to_numeric(df_cases['case:RequestedAmount'], errors='coerce').fillna(0)
    
    # num_events
    if 'num_events' in df_cases.columns:
        numeric_features.append('num_events')
    
    # num_unique_resources
    if 'num_unique_resources' in df_cases.columns:
        numeric_features.append('num_unique_resources')
    
    # duration_days
    numeric_features.append('duration_days')
    
    # Features categóricas (one-hot encoding)
    categorical_features = []
    if 'case:ApplicationType' in df_cases.columns:
        categorical_features.append('case:ApplicationType')
    if 'case:LoanGoal' in df_cases.columns:
        categorical_features.append('case:LoanGoal')
    
    # One-hot encoding para categóricas (limitado a evitar demasiadas columnas)
    for cat_feat in categorical_features:
        if cat_feat in df_cases.columns:
            # Limitar a top N categorías más frecuentes
            top_cats = df_cases[cat_feat].value_counts().head(10).index.tolist()
            df_cases[cat_feat] = df_cases[cat_feat].apply(
                lambda x: x if x in top_cats else 'Other'
            )
            dummies = pd.get_dummies(df_cases[cat_feat], prefix=cat_feat, dummy_na=False)
            df_cases = pd.concat([df_cases, dummies], axis=1)
            numeric_features.extend(dummies.columns.tolist())
            # Eliminar columna original después de one-hot
            if cat_feat in df_cases.columns:
                df_cases = df_cases.drop(columns=[cat_feat])
    
    # Seleccionar solo features numéricas para el modelo
    # Primero, asegurarse de que todas las columnas sean numéricas
    available_features = []
    for feat in numeric_features:
        if feat in df_cases.columns:
            # Intentar convertir a numérico
            try:
                df_cases[feat] = pd.to_numeric(df_cases[feat], errors='coerce')
                # Verificar que la columna sea realmente numérica
                if df_cases[feat].dtype in ['int64', 'float64', 'int32', 'float32']:
                    available_features.append(feat)
                else:
                    logger.warning(f"   ⚠️  Columna {feat} no es numérica, omitiendo")
            except Exception as e:
                logger.warning(f"   ⚠️  Error convirtiendo {feat} a numérico: {e}, omitiendo")
    
    # Crear DataFrame final solo con features numéricas válidas
    df_features = df_cases[[case_col] + available_features].copy()
    
    # Rellenar NaN y asegurar que todas sean numéricas
    for feat in available_features:
        df_features[feat] = pd.to_numeric(df_features[feat], errors='coerce').fillna(0)
    
    logger.info(f"✅ Features preparadas: {len(available_features)} features numéricas")
    logger.info(f"   Features: {', '.join(available_features[:10])}{'...' if len(available_features) > 10 else ''}")
    
    return df_features, case_col


def train_wtt_model(
    df_train_features: pd.DataFrame,
    df_train_labels: pd.DataFrame,
    case_col: str
) -> tuple:
    """
    Entrena un modelo CausalForest (WhenToTreat) sobre los datos de entrenamiento.
    
    Returns:
        Tupla (model, scaler) con el modelo entrenado y el scaler usado
    """
    logger.info("🌲 Entrenando CausalForest (WhenToTreat)...")
    
    # Preparar datos - solo columnas numéricas (EXCLUYENDO treatment y outcome para evitar data leakage)
    exclude_cols = [case_col, 'treatment_observed', 'outcome_observed', 'case_id']
    all_cols = [c for c in df_train_features.columns if c not in exclude_cols]
    feature_cols = []
    for col in all_cols:
        # Verificar que la columna sea numérica
        if df_train_features[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
            # Intentar convertir a numérico si no lo es
            try:
                df_train_features[col] = pd.to_numeric(df_train_features[col], errors='coerce')
                if df_train_features[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
                    feature_cols.append(col)
            except:
                logger.warning(f"   ⚠️  Omitiendo columna {col} (no numérica)")
        else:
            logger.warning(f"   ⚠️  Omitiendo columna {col} (tipo: {df_train_features[col].dtype})")
    
    if not feature_cols:
        raise ValueError("No hay features numéricas disponibles para entrenar el modelo")
    
    logger.info(f"   Usando {len(feature_cols)} features numéricas: {', '.join(feature_cols)}")
    
    # Asegurar que todas las columnas sean numéricas
    X = df_train_features[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X = X.values
    T = df_train_labels['treatment_observed'].values
    Y = df_train_labels['outcome_observed'].values
    
    # Normalizar features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar CausalForest
    try:
        # Intentar usar la clase de WhenToTreat si está disponible
        try:
            from causal_estimators.wtt_estimator import wtt_estimator
            
            # Crear args simulados
            class Args:
                estimator = 'CausalForest'
                propensity_model = 'LogisticRegression'
                outcome_model = 'RandomForestClassifier'
                conf_thresh = 0.1
            
            args = Args()
            wtt = wtt_estimator(args)
            wtt.initialize_forest()
            wtt.fit_forest(X_scaled, T, Y)
            model = wtt.estimator
            logger.info("✅ Usando CausalForest de WhenToTreat")
        except (ImportError, AttributeError):
            # Usar econml directamente
            import econml.grf
            model = econml.grf.CausalForest(
                n_estimators=100,
                min_samples_leaf=5,
                max_depth=None,
                random_state=42
            )
            model.fit(X_scaled, T, Y)
            logger.info("✅ Usando CausalForest de econml directamente")
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        raise
    
    logger.info("✅ Modelo entrenado exitosamente")
    
    return model, scaler, feature_cols


def apply_wtt_policy(
    df_test_features: pd.DataFrame,
    model: Any,
    scaler: MinMaxScaler,
    feature_cols: list,
    case_col: str,
    conf_threshold: float = 0.0
) -> tuple:
    """
    Aplica la política de WhenToTreat: intervenir si TE >= threshold.
    
    Args:
        conf_threshold: Threshold de confidence para intervenir (default: 0.0 = intervenir si TE > 0)
    
    Returns:
        Tupla (case_decisions, latency_ms) con las decisiones y la latencia en milisegundos
    """
    logger.info(f"🔮 Generando predicciones con threshold={conf_threshold}...")
    
    # Preparar features
    X = df_test_features[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Medir tiempo de inferencia
    start_time = time.time()
    
    # Estimar Treatment Effects
    try:
        if hasattr(model, 'estimate_ite'):
            # Cuando se usa wtt_estimator (wrapper de WhenToTreat)
            te = model.estimate_ite(w=X_scaled)
        elif hasattr(model, 'estimate_ite_forest'):
            # Cuando se usa el wrapper de WhenToTreat directamente
            te = model.estimate_ite_forest(w=X_scaled)
        elif hasattr(model, 'effect'):
            # Cuando se usa econml directamente
            te = model.effect(X_scaled)
        elif hasattr(model, 'predict'):
            # Fallback: usar predict (puede no ser TE, pero es mejor que nada)
            te = model.predict(X_scaled)
            if len(te.shape) > 1:
                te = te.flatten()
        else:
            raise AttributeError("Modelo no tiene método para estimar treatment effects")
    except Exception as e:
        logger.error(f"Error estimando treatment effects: {e}")
        logger.error(f"   Tipo de modelo: {type(model)}")
        logger.error(f"   Métodos disponibles: {[m for m in dir(model) if not m.startswith('_')]}")
        # Fallback: usar predicción simple (asumir TE = 0 para todos)
        te = np.zeros(len(X_scaled))
        logger.warning("⚠️  Usando TE=0 como fallback")
    
    # Calcular latencia total (tiempo total / número de casos)
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000  # Convertir a milisegundos
    latency_ms = total_time_ms / len(X_scaled) if len(X_scaled) > 0 else 0.0
    
    # Aplicar política: intervenir si TE >= threshold
    action_model = (te >= conf_threshold).astype(int)
    
    # Crear DataFrame con decisiones
    case_decisions = pd.DataFrame({
        case_col: df_test_features[case_col].values,
        'action_model': action_model,
        'treatment_effect': te,
        'uplift_score': te  # Para AUC-Qini
    })
    
    logger.info(f"✅ Predicciones generadas: {action_model.sum()}/{len(action_model)} casos con intervención")
    logger.info(f"⏱️  Latencia promedio: {latency_ms:.4f} ms por caso (total: {total_time_ms:.2f} ms para {len(X_scaled)} casos)")
    
    return case_decisions, latency_ms


def main():
    parser = argparse.ArgumentParser(description='Evaluación WhenToTreat para BPI Challenge 2017')
    parser.add_argument('--test', action='store_true', 
                       help='Usar archivos procesados de train/test (bpi2017_train.csv y bpi2017_test.csv)')
    args = parser.parse_args()
    
    logger.info(f"{'='*80}\nEVALUACIÓN FINAL: WhenToTreat vs BASELINE\n{'='*80}")
    
    # 1. Determinar rutas de datos
    if args.test:
        # Usar archivos procesados de train/test
        train_path = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_train.csv")
        test_path = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_test.csv")
        
        if not os.path.exists(train_path):
            logger.error(f"❌ No se encontró el archivo de train: {train_path}")
            return
        if not os.path.exists(test_path):
            logger.error(f"❌ No se encontró el archivo de test: {test_path}")
            return
        
        logger.info(f"🎯 Modo TEST: Usando archivos procesados")
        logger.info(f"📂 Train: {train_path}")
        logger.info(f"📂 Test: {test_path}")
        
        # Cargar datos de train y test por separado
        logger.info("📂 Cargando datos de entrenamiento...")
        df_train_events = load_bpi2017_data(train_path)
        logger.info("📂 Cargando datos de test...")
        df_test_events = load_bpi2017_data(test_path)
        
        # Preparar features y resultados para train
        df_train_results = prepare_baseline_dataframe(df_train_events)
        df_train_features, case_col = prepare_wtt_features(df_train_events)
        
        # Preparar features y resultados para test
        df_test_results = prepare_baseline_dataframe(df_test_events)
        df_test_features, _ = prepare_wtt_features(df_test_events)
        
        # Merge features con resultados para train
        df_train_features = df_train_features.merge(
            df_train_results[['case_id', 'treatment_observed', 'outcome_observed']],
            left_on=case_col,
            right_on='case_id',
            how='inner'
        )
        
        # Merge features con resultados para test
        df_test_features = df_test_features.merge(
            df_test_results[['case_id', 'treatment_observed', 'outcome_observed']],
            left_on=case_col,
            right_on='case_id',
            how='inner'
        )
        
        df_train = df_train_features.copy()
        df_test = df_test_features.copy()
        
        logger.info(f"📊 Train: {len(df_train)} casos para entrenamiento")
        logger.info(f"📊 Test: {len(df_test)} casos para evaluación")
        
    else:
        # Lógica original: usar config y hacer split interno
        config = load_config()
        log_path = config["log_config"]["log_path"]
        if not os.path.isabs(log_path):
            log_path = os.path.join(project_root, log_path)
        
        logger.info(f"📂 Cargando datos desde: {log_path}")
        df_events = load_bpi2017_data(log_path)
        
        # Preparar DataFrame Base (Ground Truth & Propensity)
        df_results = prepare_baseline_dataframe(df_events)
        
        # Preparar Features para WhenToTreat
        df_features, case_col = prepare_wtt_features(df_events)
        
        # Merge features con resultados para tener treatment y outcome (solo para labels, no como features)
        df_features = df_features.merge(
            df_results[['case_id', 'treatment_observed', 'outcome_observed']],
            left_on=case_col,
            right_on='case_id',
            how='inner'
        )
        
        # Split train/test (temporal: primeros 80% para train, últimos 20% para test)
        df_features = df_features.sort_values(case_col)
        n_train = int(len(df_features) * 0.8)
        df_train = df_features.iloc[:n_train].copy()
        df_test = df_features.iloc[n_train:].copy()
        
        df_test_results = df_results[df_results['case_id'].isin(df_test[case_col].values)].copy()
        
        logger.info(f"📊 Split: {len(df_train)} casos para entrenamiento, {len(df_test)} casos para evaluación")
    
    # Ruta para guardar/cargar modelo entrenado
    model_dir = os.path.join(project_root, "results/benchmark/whentotreat")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "wtt_model.pkl")
    scaler_path = os.path.join(model_dir, "wtt_scaler.pkl")
    features_path = os.path.join(model_dir, "wtt_features.pkl")
    
    # 6. Entrenar o cargar modelo
    model = None
    scaler = None
    feature_cols = None
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        logger.info("📥 Cargando modelo pre-entrenado...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_cols = pickle.load(f)
            logger.info("✅ Modelo cargado exitosamente")
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}. Reentrenando...")
            model = None
    
    if model is None:
        logger.info("🏋️  Entrenando nuevo modelo...")
        # Separar features de labels
        df_train_features_only = df_train[[case_col] + [c for c in df_train.columns if c not in ['case_id', 'treatment_observed', 'outcome_observed']]].copy()
        df_train_labels = df_train[['treatment_observed', 'outcome_observed']].copy()
        model, scaler, feature_cols = train_wtt_model(
            df_train_features_only,
            df_train_labels,
            case_col
        )
        
        # Guardar modelo
        logger.info("💾 Guardando modelo entrenado...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(features_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        logger.info("✅ Modelo guardado")
    
    # 7. Aplicar política de WhenToTreat
    # Usar threshold=0.0 por defecto (intervenir si TE > 0)
    # Se puede optimizar más adelante
    conf_threshold = 0.0
    model_decisions, latency_ms = apply_wtt_policy(
        df_test,
        model,
        scaler,
        feature_cols,
        case_col,
        conf_threshold=conf_threshold
    )
    
    # 8. Merge decisiones del modelo en df_results
    # Si usamos archivos procesados, df_test_results ya está preparado
    if not args.test:
        # Filtrar df_results para solo casos de test (evitar data leakage)
        test_case_ids = df_test[case_col].values
        df_test_results = df_results[df_results['case_id'].isin(test_case_ids)].copy()
    
    if 'action_model' in df_test_results.columns:
        del df_test_results['action_model']
    
    df_final = df_test_results.merge(
        model_decisions.rename(columns={case_col: 'case_id'}),
        on='case_id',
        how='left'
    )
    # Para casos que no están en df_test (por si hay diferencias), usar 0 (no intervenir)
    df_final['action_model'] = df_final['action_model'].fillna(0).astype(int)
    
    # Agregar uplift_score si no existe
    if 'uplift_score' not in df_final.columns and 'treatment_effect' in df_final.columns:
        df_final['uplift_score'] = df_final['treatment_effect']
    
    # 9. Calcular Métricas con BenchmarkEvaluator
    logger.info("📊 Calculando métricas de rendimiento...")
    evaluator = BenchmarkEvaluator()
    metrics = evaluator.evaluate(df_final, training_complexity="Media (CPU - Forest)")
    
    # Agregar latencia a las métricas
    metrics['latency_ms'] = latency_ms
    
    # 10. Comparar con Baseline
    # Si usamos archivos procesados, buscar baseline en bpi2017_test
    if args.test:
        baseline_csv = os.path.join(project_root, "results/benchmark/bpi2017_test/baseline_metrics.csv")
    else:
        baseline_csv = os.path.join(project_root, "results/benchmark/bpi-challenge-2017-sample/baseline_metrics.csv")
        if not os.path.exists(baseline_csv):
            baseline_csv = os.path.join(project_root, "results/benchmark/baseline_metrics.csv")
    
    baseline_gain = 0.0
    if os.path.exists(baseline_csv):
        try:
            base_df = pd.read_csv(baseline_csv)
            baseline_gain = base_df['net_gain'].iloc[0]
            logger.info(f"📉 Baseline Net Gain cargado desde {baseline_csv}: ${baseline_gain:.2f}")
        except Exception as e:
            logger.warning(f"No se pudo leer baseline_metrics.csv: {e}, asumiendo 0 para Lift")
    else:
        logger.warning(f"Archivo baseline_metrics.csv no encontrado en: {baseline_csv}")
    
    # Recalcular Lift real
    model_gain = metrics['net_gain']
    if baseline_gain != 0:
        real_lift = ((model_gain - baseline_gain) / abs(baseline_gain)) * 100
        metrics['lift_vs_bau'] = real_lift
    
    # 11. Reporte Final
    print("\n" + "="*60)
    print("🏆 RESULTADOS FINALES: WhenToTreat (CausalForest)")
    print("="*60)
    print(f"💰 Net Gain (Modelo):     ${metrics['net_gain']:.2f}")
    print(f"📉 Baseline Gain:         ${baseline_gain:.2f}")
    print(f"🚀 LIFT REAL:             {metrics['lift_vs_bau']:+.2f}%")
    print(f"🎯 % Intervenciones:      {metrics['intervention_percentage']:.2f}%")
    print(f"🛡️  % Violaciones:         {metrics['violation_percentage']:.2f}%")
    if metrics.get('auc_qini') is not None:
        print(f"📈 AUC-Qini:              {metrics['auc_qini']:.4f}")
    if metrics.get('latency_ms') is not None:
        print(f"⏱️  Latencia:              {metrics['latency_ms']:.4f} ms/caso")
    print("="*60 + "\n")
    
    # 12. Guardar
    if args.test:
        out_dir = os.path.join(project_root, "results/benchmark/bpi2017_test")
    else:
        out_dir = os.path.join(project_root, "results/benchmark/whentotreat")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "whentotreat_metrics.csv"), index=False)
    logger.info(f"✅ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()

