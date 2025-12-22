#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUACIÓN FINAL: Cost-Aware Cycle Time Reduction vs BASELINE
---------------------------------------------------------------
Este script entrena/evalúa el modelo Cost-Aware (DMLOrthoForest) sobre el log histórico.
Genera predicciones basadas en Treatment Effects y calcula el Net Gain usando IPW.
Política: Intervenir si TE < threshold (reducir duración).
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from typing import Optional, Dict, Any, Tuple, List

# Agregar directorios al path (IMPORTANTE: hacer esto PRIMERO para evitar conflictos)
script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark
src_dir = os.path.dirname(script_dir)  # src/
project_root = os.path.dirname(src_dir)  # root
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar utils del proyecto ACTUAL primero
from utils.config import load_config
from utils.logger_utils import setup_logger
from benchmark_evaluator import BenchmarkEvaluator
from test_baseline_bpi2017 import load_bpi2017_data, estimate_propensity_score, prepare_baseline_dataframe

# Intentar importar econml
try:
    from econml.orf import DMLOrthoForest
    from econml.sklearn_extensions.linear_model import WeightedLasso
    econml_available = True
    logger = setup_logger(__name__)
    logger.info("✅ econml disponible")
except ImportError:
    logger = setup_logger(__name__)
    logger.error("❌ econml no está instalado. Instala con: pip install econml")
    sys.exit(1)

logger = setup_logger(__name__)


def prepare_cost_aware_features(df_events: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Prepara features para Cost-Aware Cycle Time Reduction desde el log de eventos.
    Basado en bpi2017_experiments.py del repositorio cost-aware.
    
    Returns:
        Tupla (df_features, case_col) con features preparadas y nombre de columna de caso
    """
    logger.info("🔧 Preparando features para Cost-Aware Cycle Time Reduction...")
    
    # Identificar columnas
    case_col = 'case:concept:name' if 'case:concept:name' in df_events.columns else df_events.columns[0]
    act_col = 'concept:name' if 'concept:name' in df_events.columns else df_events.columns[1]
    time_col = 'time:timestamp' if 'time:timestamp' in df_events.columns else None
    resource_col = 'org:resource' if 'org:resource' in df_events.columns else None
    
    # Convertir timestamp si existe
    if time_col and time_col in df_events.columns:
        df_events[time_col] = pd.to_datetime(df_events[time_col], utc=True, format='mixed', errors='coerce')
    
    # Agrupar por caso para obtener features agregadas
    df_cases = df_events.groupby(case_col).agg({
        act_col: 'count',  # num_events
        time_col: ['min', 'max'] if time_col else None
    }).reset_index()
    
    # Aplanar columnas multi-nivel
    df_cases.columns = [case_col] + [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_cases.columns[1:]]
    
    # Renombrar columnas
    if f"{act_col}_count" in df_cases.columns:
        df_cases = df_cases.rename(columns={f"{act_col}_count": "num_events"})
    if time_col and f"{time_col}_min" in df_cases.columns:
        df_cases['duration_days'] = (
            pd.to_datetime(df_cases[f"{time_col}_max"]) - 
            pd.to_datetime(df_cases[f"{time_col}_min"])
        ).dt.total_seconds() / (24 * 3600)
        df_cases = df_cases.drop(columns=[f"{time_col}_min", f"{time_col}_max"])
    else:
        df_cases['duration_days'] = 0.0
    
    # Features numéricas estáticas
    static_numeric = ['case:RequestedAmount', 'RequestedAmount']
    for col in static_numeric:
        if col in df_events.columns:
            case_feat = df_events.groupby(case_col)[col].first().reset_index()
            df_cases = df_cases.merge(case_feat, on=case_col, how='left')
            if col != 'case:RequestedAmount':
                df_cases = df_cases.rename(columns={col: 'case:RequestedAmount'})
            break
    
    # Features categóricas estáticas
    static_categorical = ['case:ApplicationType', 'case:LoanGoal']
    for col in static_categorical:
        if col in df_events.columns:
            case_feat = df_events.groupby(case_col)[col].first().reset_index()
            df_cases = df_cases.merge(case_feat, on=case_col, how='left')
    
    # Features numéricas básicas
    numeric_features = []
    if 'case:RequestedAmount' in df_cases.columns:
        df_cases['case:RequestedAmount'] = pd.to_numeric(df_cases['case:RequestedAmount'], errors='coerce').fillna(0)
        numeric_features.append('case:RequestedAmount')
    if 'num_events' in df_cases.columns:
        numeric_features.append('num_events')
    if 'duration_days' in df_cases.columns:
        numeric_features.append('duration_days')
    
    # Features categóricas (one-hot encoding)
    categorical_features = []
    for col in static_categorical:
        if col in df_cases.columns:
            categorical_features.append(col)
    
    # One-hot encoding para categóricas
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
    
    # Seleccionar solo features numéricas válidas
    available_features = []
    for feat in numeric_features:
        if feat in df_cases.columns:
            try:
                df_cases[feat] = pd.to_numeric(df_cases[feat], errors='coerce')
                if df_cases[feat].dtype in ['int64', 'float64', 'int32', 'float32']:
                    available_features.append(feat)
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


def train_cost_aware_model(
    df_train_features: pd.DataFrame,
    df_train_labels: pd.DataFrame,
    case_col: str
) -> Tuple[Any, StandardScaler, StandardScaler, List[str], List[str], List[str], List[str]]:
    """
    Entrena un modelo DMLOrthoForest (Cost-Aware) sobre los datos de entrenamiento.
    
    Returns:
        Tupla (model, scaler_X, scaler_W, num_cols, cat_cols, num_cols, cat_cols) 
        con el modelo, scalers y listas de columnas
    """
    logger.info("🌲 Entrenando DMLOrthoForest (Cost-Aware)...")
    
    # Preparar datos - solo columnas numéricas (EXCLUYENDO treatment y outcome para evitar data leakage)
    # NOTA: duration_days puede usarse como feature (es el outcome histórico, no el futuro)
    exclude_cols = [case_col, 'treatment_observed', 'outcome_observed', 'case_id']
    all_cols = [c for c in df_train_features.columns if c not in exclude_cols]
    feature_cols = []
    for col in all_cols:
        # Verificar que la columna sea numérica
        if df_train_features[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
            try:
                df_train_features[col] = pd.to_numeric(df_train_features[col], errors='coerce')
                if df_train_features[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
                    feature_cols.append(col)
            except:
                logger.warning(f"   ⚠️  Omitiendo columna {col} (no numérica)")
        else:
            # Intentar convertir a numérico
            try:
                df_train_features[col] = pd.to_numeric(df_train_features[col], errors='coerce')
                if df_train_features[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
                    feature_cols.append(col)
            except:
                logger.warning(f"   ⚠️  Omitiendo columna {col} (tipo: {df_train_features[col].dtype})")
    
    if not feature_cols:
        logger.error(f"   Columnas disponibles: {list(df_train_features.columns)}")
        logger.error(f"   Columnas excluidas: {exclude_cols}")
        raise ValueError("No hay features numéricas disponibles para entrenar el modelo")
    
    # Separar features en numéricas y categóricas (one-hot encoded)
    # Basado en bpi2017_experiments.py: cat_confound_cols = cat_hetero_cols = ['LoanGoal', 'ApplicationType']
    cat_cols = [c for c in feature_cols if any(cat in c for cat in ['ApplicationType', 'LoanGoal'])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    logger.info(f"   Features numéricas: {len(num_cols)}")
    logger.info(f"   Features categóricas (one-hot): {len(cat_cols)}")
    
    # Preparar X (heterogeneity features): numéricas escaladas + categóricas one-hot
    if not num_cols:
        logger.error(f"   Columnas en df_train_features: {list(df_train_features.columns)}")
        logger.error(f"   num_cols: {num_cols}")
        logger.error(f"   cat_cols: {cat_cols}")
        raise ValueError("No hay features numéricas disponibles")
    
    logger.info(f"   Preparando X con {len(num_cols)} features numéricas: {num_cols}")
    X_num = df_train_features[num_cols].values
    logger.info(f"   X_num shape: {X_num.shape}")
    
    scaler_X = StandardScaler()
    X_num_scaled = scaler_X.fit_transform(X_num)
    logger.info(f"   X_num_scaled shape: {X_num_scaled.shape}")
    
    if cat_cols:
        X_cat = df_train_features[cat_cols].values
        X = np.concatenate([X_num_scaled, X_cat], axis=1)
        logger.info(f"   X shape (con cat): {X.shape}")
    else:
        X = X_num_scaled
        logger.info(f"   X shape (sin cat): {X.shape}")
    
    # Preparar W (confounders): mismas features que X (según código original)
    W_num = df_train_features[num_cols].values
    scaler_W = StandardScaler()
    W_num_scaled = scaler_W.fit_transform(W_num)
    
    if cat_cols:
        W_cat = df_train_features[cat_cols].values
        W = np.concatenate([W_num_scaled, W_cat], axis=1)
    else:
        W = W_num_scaled
    
    # Preparar Y (outcome: duration_days) y T (treatment)
    Y = df_train_labels['duration_days'].values  # Outcome es duración
    T = df_train_labels['treatment_observed'].values
    
    # Asegurar que Y y T sean arrays 1D (usar ravel() para evitar warnings de sklearn)
    Y = np.ravel(Y)
    T = np.ravel(T)
    
    # Para discrete_treatment=True, T debe ser (n_samples,) o (n_samples, 1)
    # Asegurar que T sea 1D
    if T.ndim > 1:
        T = T.flatten()
    
    logger.info(f"   Y shape: {Y.shape}, T shape: {T.shape}")
    logger.info(f"   X shape: {X.shape}, W shape: {W.shape}")
    logger.info(f"   Y range: [{Y.min():.2f}, {Y.max():.2f}], T unique: {np.unique(T)}")
    
    # Verificar que T tenga al menos 2 valores únicos (requisito para discrete_treatment=True)
    t_unique = np.unique(T)
    if len(t_unique) < 2:
        logger.error(f"❌ T tiene solo un valor único ({t_unique}).")
        logger.error("   No se puede entrenar un modelo causal sin variación en el tratamiento.")
        logger.error("   Esto puede ocurrir en datasets muy pequeños o cuando todos los casos tienen tratamiento.")
        raise ValueError(f"No se puede entrenar modelo causal: T tiene solo un valor único ({t_unique}). Se requiere al menos 2 valores únicos para discrete_treatment=True.")
    
    # Entrenar DMLOrthoForest
    # Parámetros basados en bpi2017_experiments.py
    n_trees = 200
    min_leaf_size = 20
    max_depth = 30
    subsample_ratio = 0.4
    lambda_reg = 0.01
    
    try:
        # Ajustar min_leaf_size si el dataset es muy pequeño
        if len(Y) < min_leaf_size * 2:
            min_leaf_size = max(1, len(Y) // 4)
            logger.info(f"   Ajustando min_leaf_size a {min_leaf_size} para dataset pequeño")
        
        model = DMLOrthoForest(
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            subsample_ratio=subsample_ratio,
            discrete_treatment=True,
            model_T=LogisticRegression(C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga'),
            model_Y=Lasso(alpha=lambda_reg),
            model_T_final=LogisticRegression(C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga'),
            model_Y_final=WeightedLasso(alpha=lambda_reg),
            random_state=42
        )
        logger.info(f"   Llamando model.fit(Y, T, X=X, W=W)...")
        # Verificar que X y W no estén vacíos
        if X.shape[1] == 0:
            raise ValueError(f"X está vacío (shape: {X.shape})")
        if W.shape[1] == 0:
            raise ValueError(f"W está vacío (shape: {W.shape})")
        
        model.fit(Y, T, X=X, W=W)
        logger.info("✅ Modelo DMLOrthoForest entrenado exitosamente")
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        logger.error(f"   Y shape: {Y.shape if 'Y' in locals() else 'N/A'}")
        logger.error(f"   T shape: {T.shape if 'T' in locals() else 'N/A'}")
        logger.error(f"   X shape: {X.shape if 'X' in locals() else 'N/A'}")
        logger.error(f"   W shape: {W.shape if 'W' in locals() else 'N/A'}")
        logger.error(f"   T unique values: {np.unique(T) if 'T' in locals() else 'N/A'}")
        raise
    
    return model, scaler_X, scaler_W, num_cols, cat_cols, num_cols, cat_cols


def apply_cost_aware_policy(
    df_test_features: pd.DataFrame,
    model: Any,
    scaler_X: StandardScaler,
    scaler_W: StandardScaler,
    num_cols_X: List[str],
    cat_cols_X: List[str],
    num_cols_W: List[str],
    cat_cols_W: List[str],
    case_col: str,
    conf_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Aplica la política de Cost-Aware: intervenir si TE < threshold (reducir duración).
    
    Args:
        conf_threshold: Threshold de confidence para intervenir (default: 0.0 = intervenir si TE < 0)
    """
    logger.info(f"🔮 Generando predicciones con threshold={conf_threshold}...")
    
    # Preparar features X (heterogeneity)
    X_num = df_test_features[num_cols_X].values
    X_num_scaled = scaler_X.transform(X_num)
    
    if cat_cols_X:
        X_cat = df_test_features[cat_cols_X].values
        X = np.concatenate([X_num_scaled, X_cat], axis=1)
    else:
        X = X_num_scaled
    
    # Preparar features W (confounders)
    W_num = df_test_features[num_cols_W].values
    W_num_scaled = scaler_W.transform(W_num)
    
    if cat_cols_W:
        W_cat = df_test_features[cat_cols_W].values
        W = np.concatenate([W_num_scaled, W_cat], axis=1)
    else:
        W = W_num_scaled
    
    # Estimar Treatment Effects
    try:
        # DMLOrthoForest usa const_marginal_effect (no necesita W para predicción)
        # Dividir en batches para evitar problemas de memoria
        batch_size = 100
        if len(X) <= batch_size:
            te = model.const_marginal_effect(X)
        else:
            batches = np.array_split(X, max(1, len(X) // batch_size))
            te_list = []
            for batch in batches:
                estimates = model.const_marginal_effect(batch)
                te_list.append(estimates)
            te = np.concatenate(te_list, axis=0)
        
        # Aplanar a 1D si es necesario (discrete_treatment=True puede devolver (n_samples, n_treatments))
        if te.ndim > 1:
            # Si hay múltiples tratamientos, tomar el primero (o promedio)
            # Para binary treatment, debería ser (n_samples, 1) o (n_samples,)
            if te.shape[1] == 1:
                te = te.flatten()
            else:
                # Si hay múltiples tratamientos, usar el primero
                logger.warning(f"   TE tiene forma {te.shape}, usando primera columna")
                te = te[:, 0]
        
        # Asegurar que es 1D
        te = te.flatten() if te.ndim > 1 else te
        
    except Exception as e:
        logger.error(f"Error estimando treatment effects: {e}")
        logger.error(f"   Tipo de modelo: {type(model)}")
        logger.error(f"   Métodos disponibles: {[m for m in dir(model) if not m.startswith('_')]}")
        # Fallback: usar predicción simple (asumir TE = 0 para todos)
        te = np.zeros(len(X))
        logger.warning("⚠️  Usando TE=0 como fallback")
    
    # Asegurar que te es 1D y tiene la longitud correcta
    if te.ndim > 1:
        te = te.flatten()
    if len(te) != len(X):
        logger.warning(f"   TE tiene longitud {len(te)} pero X tiene {len(X)}, ajustando...")
        if len(te) > len(X):
            te = te[:len(X)]
        else:
            te = np.pad(te, (0, len(X) - len(te)), mode='constant', constant_values=0)
    
    # Aplicar política: intervenir si TE < threshold (porque queremos reducir duración)
    # TE negativo = tratamiento reduce duración = debemos intervenir
    action_model = (te < conf_threshold).astype(int)
    
    # Crear DataFrame con decisiones (asegurar que todos los arrays sean 1D)
    case_decisions = pd.DataFrame({
        case_col: df_test_features[case_col].values.flatten() if df_test_features[case_col].values.ndim > 1 else df_test_features[case_col].values,
        'action_model': action_model.flatten() if action_model.ndim > 1 else action_model,
        'treatment_effect': te.flatten() if te.ndim > 1 else te,
        'uplift_score': (-te).flatten() if te.ndim > 1 else -te  # Negativo porque queremos reducir duración (TE negativo es bueno)
    })
    
    logger.info(f"✅ Predicciones generadas: {action_model.sum()}/{len(action_model)} casos con intervención")
    
    return case_decisions


def main():
    logger.info(f"{'='*80}\nEVALUACIÓN FINAL: Cost-Aware Cycle Time Reduction vs BASELINE\n{'='*80}")
    
    # 1. Configuración y Rutas
    config = load_config()
    log_path = config["log_config"]["log_path"]
    if not os.path.isabs(log_path):
        log_path = os.path.join(project_root, log_path)
    
    # Ruta para guardar/cargar modelo entrenado
    model_dir = os.path.join(project_root, "results/benchmark/cost_aware")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cost_aware_model.pkl")
    scaler_X_path = os.path.join(model_dir, "cost_aware_scaler_X.pkl")
    scaler_W_path = os.path.join(model_dir, "cost_aware_scaler_W.pkl")
    features_X_path = os.path.join(model_dir, "cost_aware_features_X.pkl")
    features_W_path = os.path.join(model_dir, "cost_aware_features_W.pkl")
    
    # 2. Cargar Datos
    logger.info(f"📂 Cargando datos desde: {log_path}")
    df_events = load_bpi2017_data(log_path)
    
    # 3. Preparar DataFrame Base (Ground Truth & Propensity)
    df_results = prepare_baseline_dataframe(df_events)
    
    # 4. Preparar Features para Cost-Aware
    df_features, case_col = prepare_cost_aware_features(df_events)
    
    # Merge features con resultados para tener treatment y outcome
    # Nota: duration_days puede estar en df_features o en df_results, usar el de df_results si existe
    merge_cols = ['case_id', 'treatment_observed', 'outcome_observed']
    if 'duration_days' in df_results.columns:
        merge_cols.append('duration_days')
    
    df_features = df_features.merge(
        df_results[merge_cols],
        left_on=case_col,
        right_on='case_id',
        how='inner',
        suffixes=('', '_from_results')
    )
    
    # Si duration_days no estaba en df_features, usar el de df_results
    if 'duration_days' not in df_features.columns and 'duration_days_from_results' in df_features.columns:
        df_features['duration_days'] = df_features['duration_days_from_results']
        df_features = df_features.drop(columns=['duration_days_from_results'])
    elif 'duration_days' not in df_features.columns:
        # Si no está en ninguno, calcular desde df_results
        df_features = df_features.merge(
            df_results[['case_id', 'duration_days']],
            left_on=case_col,
            right_on='case_id',
            how='left',
            suffixes=('', '_from_results2')
        )
        if 'duration_days_from_results2' in df_features.columns:
            df_features['duration_days'] = df_features['duration_days_from_results2']
            df_features = df_features.drop(columns=['duration_days_from_results2'])
    
    # Limpiar columnas duplicadas del merge
    if 'case_id_from_results' in df_features.columns:
        df_features = df_features.drop(columns=['case_id_from_results'])
    
    # Guardar labels separadamente (no deben usarse como features)
    required_cols = ['case_id', 'treatment_observed', 'outcome_observed']
    if 'duration_days' in df_features.columns:
        required_cols.append('duration_days')
    df_labels = df_features[required_cols].copy()
    
    # 5. Split train/test (temporal: primeros 80% para train, últimos 20% para test)
    df_features = df_features.sort_values(case_col)
    n_train = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:n_train].copy()
    df_test = df_features.iloc[n_train:].copy()
    
    logger.info(f"📊 Split: {len(df_train)} casos para entrenamiento, {len(df_test)} casos para evaluación")
    logger.info("   (Nota: Evaluación sobre conjunto de test para evitar data leakage)")
    
    # 6. Entrenar o cargar modelo
    model = None
    scaler_X = None
    scaler_W = None
    num_cols_X = None
    cat_cols_X = None
    num_cols_W = None
    cat_cols_W = None
    
    if all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_W_path, features_X_path, features_W_path]):
        logger.info("📥 Cargando modelo pre-entrenado...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_X_path, 'rb') as f:
                scaler_X = pickle.load(f)
            with open(scaler_W_path, 'rb') as f:
                scaler_W = pickle.load(f)
            with open(features_X_path, 'rb') as f:
                features_data = pickle.load(f)
                num_cols_X = features_data.get('num_cols', [])
                cat_cols_X = features_data.get('cat_cols', [])
            with open(features_W_path, 'rb') as f:
                features_data = pickle.load(f)
                num_cols_W = features_data.get('num_cols', [])
                cat_cols_W = features_data.get('cat_cols', [])
            logger.info("✅ Modelo cargado exitosamente")
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}. Reentrenando...")
            model = None
    
    if model is None:
        logger.info("🏋️  Entrenando nuevo modelo...")
        # Separar features de labels
        # NOTA: duration_days puede usarse como feature (es histórico, no futuro)
        exclude_from_features = ['case_id', 'treatment_observed', 'outcome_observed']
        feature_cols_to_use = [c for c in df_train.columns if c not in exclude_from_features and c != case_col]
        df_train_features_only = df_train[[case_col] + feature_cols_to_use].copy()
        df_train_labels = df_train[['treatment_observed', 'outcome_observed', 'duration_days']].copy()
        
        logger.info(f"   Columnas disponibles para features: {feature_cols_to_use}")
        model, scaler_X, scaler_W, num_cols_X, cat_cols_X, num_cols_W, cat_cols_W = train_cost_aware_model(
            df_train_features_only,
            df_train_labels,
            case_col
        )
        
        # Guardar modelo
        logger.info("💾 Guardando modelo entrenado...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(scaler_W_path, 'wb') as f:
            pickle.dump(scaler_W, f)
        with open(features_X_path, 'wb') as f:
            pickle.dump({'num_cols': num_cols_X, 'cat_cols': cat_cols_X}, f)
        with open(features_W_path, 'wb') as f:
            pickle.dump({'num_cols': num_cols_W, 'cat_cols': cat_cols_W}, f)
        logger.info("✅ Modelo guardado")
    
    # 7. Aplicar política de Cost-Aware
    # Usar threshold=0.0 por defecto (intervenir si TE < 0, es decir, si reduce duración)
    conf_threshold = 0.0
    model_decisions = apply_cost_aware_policy(
        df_test,
        model,
        scaler_X,
        scaler_W,
        num_cols_X,
        cat_cols_X,
        num_cols_W,
        cat_cols_W,
        case_col,
        conf_threshold=conf_threshold
    )
    
    # 8. Merge decisiones del modelo en df_results
    test_case_ids = df_test[case_col].values
    df_test_results = df_results[df_results['case_id'].isin(test_case_ids)].copy()
    
    if 'action_model' in df_test_results.columns:
        del df_test_results['action_model']
    
    df_final = df_test_results.merge(
        model_decisions.rename(columns={case_col: 'case_id'}),
        on='case_id',
        how='left'
    )
    df_final['action_model'] = df_final['action_model'].fillna(0).astype(int)
    
    # Agregar uplift_score si no existe
    if 'uplift_score' not in df_final.columns and 'treatment_effect' in df_final.columns:
        df_final['uplift_score'] = df_final['treatment_effect']
    
    # 9. Calcular Métricas con BenchmarkEvaluator
    logger.info("📊 Calculando métricas de rendimiento...")
    evaluator = BenchmarkEvaluator()
    metrics = evaluator.evaluate(df_final, training_complexity="Alta (CPU - OrthoForest)")
    
    # 10. Comparar con Baseline
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
    
    # Recalcular Lift real
    model_gain = metrics['net_gain']
    if baseline_gain != 0:
        real_lift = ((model_gain - baseline_gain) / abs(baseline_gain)) * 100
        metrics['lift_vs_bau'] = real_lift
    
    # 11. Reporte Final
    print("\n" + "="*60)
    print("🏆 RESULTADOS FINALES: Cost-Aware Cycle Time Reduction")
    print("="*60)
    print(f"💰 Net Gain (Modelo):     ${metrics['net_gain']:.2f}")
    print(f"📉 Baseline Gain:         ${baseline_gain:.2f}")
    print(f"🚀 LIFT REAL:             {metrics['lift_vs_bau']:+.2f}%")
    print(f"🎯 % Intervenciones:      {metrics['intervention_percentage']:.2f}%")
    print(f"🛡️  % Violaciones:         {metrics['violation_percentage']:.2f}%")
    if metrics.get('auc_qini') is not None:
        print(f"📈 AUC-Qini:              {metrics['auc_qini']:.4f}")
    print("="*60 + "\n")
    
    # 12. Guardar
    out_dir = os.path.join(project_root, "results/benchmark/cost_aware")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "cost_aware_metrics.csv"), index=False)
    logger.info(f"✅ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()

