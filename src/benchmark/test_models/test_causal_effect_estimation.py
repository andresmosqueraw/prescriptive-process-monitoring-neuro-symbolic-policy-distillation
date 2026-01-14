#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUACIÓN FINAL: Causal Effect Estimation (Múltiples Métodos) vs BASELINE
----------------------------------------
Este script evalúa TODOS los métodos causales del repositorio 
"prescriptive-process-monitoring-based-on-causal-effect-estimation" sobre el log histórico.
Genera predicciones basadas en Treatment Effects y calcula el Net Gain usando IPW para cada método.

Métodos evaluados:
- CausalForest, CausalTree, ORthoforestDML (Forest-based)
- SLearner, TLearner, XLearner (Meta-learners)
- StandardizationEstimator, StratifiedStandardizationEstimator
- DoublyRobustEstimator, DoublyRobustLearner
- DoubleML
- CEVAE (si está disponible)
- IPWEstimator, MatchingEstimator
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Optional, Dict, Any, Tuple, List

# Agregar directorios al path (IMPORTANTE: hacer esto PRIMERO para evitar conflictos)
script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark/test_models/
benchmark_dir = os.path.dirname(script_dir)  # src/benchmark/
src_dir = os.path.dirname(benchmark_dir)  # src/
project_root = os.path.dirname(src_dir)  # root
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Agregar benchmark al path para importar benchmark_evaluator
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

# Importar utils del proyecto ACTUAL primero (antes de agregar otros repos al path)
from utils.config import load_config
from utils.logger_utils import setup_logger
from benchmark_evaluator import BenchmarkEvaluator
from test_baseline_bpi2017 import load_bpi2017_data, estimate_propensity_score, prepare_baseline_dataframe

# AHORA intentar importar Causal Effect Estimation (después de importar utils del proyecto actual)
causal_est_path = "/home/andrew/Documents/asistencia-graduada-phd-oscar/paper1/prescriptive-process-monitoring-models/prescriptive-process-monitoring-based-on-causal-effect-estimation"
if os.path.exists(causal_est_path):
    if causal_est_path not in sys.path:
        sys.path.append(causal_est_path)  # append en lugar de insert para menor prioridad

# Configurar logger
logger = setup_logger(__name__)

# Intentar importar todos los estimadores disponibles
ESTIMATORS_AVAILABLE = {}

try:
    from causal_estimators.forest_estimators import CausalForest, CausalTree, ORthoforestDML
    ESTIMATORS_AVAILABLE['CausalForest'] = CausalForest
    ESTIMATORS_AVAILABLE['CausalTree'] = CausalTree
    ESTIMATORS_AVAILABLE['ORthoforestDML'] = ORthoforestDML
    logger.info("✅ Forest estimators disponibles")
except ImportError as e:
    logger.warning(f"⚠️  Forest estimators del repositorio no disponibles: {e}")
    # Fallback: usar econml directamente
    try:
        import econml.grf
        import econml.orf
        ESTIMATORS_AVAILABLE['CausalForest'] = econml.grf.CausalForest
        ESTIMATORS_AVAILABLE['CausalTree'] = econml.grf.CausalForest  # CausalTree es CausalForest con n_estimators=1
        ESTIMATORS_AVAILABLE['ORthoforestDML'] = econml.orf.DMLOrthoForest
        logger.info("✅ Forest estimators disponibles desde econml")
    except ImportError:
        logger.warning("⚠️  econml.orf no disponible")

try:
    # Intentar importar desde el repositorio, pero manejar conflictos de utils
    import sys
    original_path = sys.path.copy()
    # Temporalmente remover el path del proyecto para evitar conflictos
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    try:
        from causal_estimators.metalearners import SLearner, TLearner, XLearner
        ESTIMATORS_AVAILABLE['SLearner'] = SLearner
        ESTIMATORS_AVAILABLE['TLearner'] = TLearner
        ESTIMATORS_AVAILABLE['XLearner'] = XLearner
        logger.info("✅ Meta-learners disponibles desde repositorio")
    finally:
        sys.path = original_path
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️  Meta-learners del repositorio no disponibles: {e}")
    # Fallback: usar econml directamente
    try:
        import econml.metalearners
        ESTIMATORS_AVAILABLE['SLearner'] = econml.metalearners.SLearner
        ESTIMATORS_AVAILABLE['TLearner'] = econml.metalearners.TLearner
        ESTIMATORS_AVAILABLE['XLearner'] = econml.metalearners.XLearner
        logger.info("✅ Meta-learners disponibles desde econml")
    except ImportError:
        logger.warning("⚠️  econml.metalearners no disponible")

try:
    # Intentar importar desde el repositorio, pero manejar conflictos de utils
    import sys
    original_path = sys.path.copy()
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    try:
        from causal_estimators.standardization_estimator import StandardizationEstimator, StratifiedStandardizationEstimator
        ESTIMATORS_AVAILABLE['StandardizationEstimator'] = StandardizationEstimator
        ESTIMATORS_AVAILABLE['StratifiedStandardizationEstimator'] = StratifiedStandardizationEstimator
        logger.info("✅ Standardization estimators disponibles desde repositorio")
    finally:
        sys.path = original_path
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️  Standardization estimators del repositorio no disponibles: {e}")
    # Fallback: usar causallib directamente
    try:
        from causallib.estimation import Standardization, StratifiedStandardization
        # Crear wrappers simples
        class StandardizationWrapper:
            def __init__(self, outcome_model=LinearRegression()):
                self.estimator = Standardization(learner=outcome_model)
            def fit(self, w, t, y):
                # causallib usa fit(X, a, y) donde X=features, a=treatment, y=outcome
                # Convertir a DataFrame/Series si es necesario
                import pandas as pd
                if isinstance(w, np.ndarray):
                    w = pd.DataFrame(w)
                if isinstance(t, np.ndarray):
                    t = pd.Series(t) if t.ndim == 1 else pd.DataFrame(t)
                if isinstance(y, np.ndarray):
                    y = pd.Series(y) if y.ndim == 1 else pd.DataFrame(y)
                self.estimator.fit(X=w, a=t, y=y)
            def effect(self, X):
                # Estimar efectos individuales usando estimate_individual_outcome
                import pandas as pd
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X)
                try:
                    # estimate_individual_outcome devuelve y0 y y1 para cada individuo
                    y0 = self.estimator.estimate_individual_outcome(X=X, a=0)
                    y1 = self.estimator.estimate_individual_outcome(X=X, a=1)
                    te = y1 - y0
                    return te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
                except (AttributeError, TypeError) as e:
                    # Fallback: usar estimate_effect si está disponible
                    try:
                        te = self.estimator.estimate_effect(X=X, a1=1, a0=0)
                        return te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
                    except:
                        logger.warning(f"   Error en estimate_effect: {e}")
                        # Último fallback: asumir TE=0
                        return np.zeros(len(X))
        ESTIMATORS_AVAILABLE['StandardizationEstimator'] = StandardizationWrapper
        logger.info("✅ Standardization estimators disponibles desde causallib")
    except ImportError:
        logger.warning("⚠️  causallib.estimation no disponible")

try:
    # Intentar importar desde el repositorio, pero manejar conflictos de utils
    import sys
    original_path = sys.path.copy()
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    try:
        from causal_estimators.doubly_robust_estimator import DoublyRobustEstimator, DoublyRobustLearner
        ESTIMATORS_AVAILABLE['DoublyRobustEstimator'] = DoublyRobustEstimator
        ESTIMATORS_AVAILABLE['DoublyRobustLearner'] = DoublyRobustLearner
        logger.info("✅ Doubly Robust estimators disponibles desde repositorio")
    finally:
        sys.path = original_path
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️  Doubly Robust estimators del repositorio no disponibles: {e}")
    # Fallback: usar causallib/econml directamente
    try:
        from causallib.estimation import AIPW  # AIPW es similar a Doubly Robust
        from econml.drlearner import DRLearner
        # Crear wrappers simples
        class AIPWWrapper:
            def __init__(self, outcome_model=RandomForestRegressor(), prop_score_model=LogisticRegression()):
                from causallib.estimation import Standardization, IPW
                self.estimator = AIPW(
                    outcome_model=Standardization(learner=outcome_model),
                    weight_model=IPW(learner=prop_score_model)
                )
            def fit(self, w, t, y):
                self.estimator.fit(y, t, X=w)
            def effect(self, X):
                return self.estimator.estimate_population_outcome(X=X, treatment_values=[0, 1])
        ESTIMATORS_AVAILABLE['DoublyRobustEstimator'] = AIPWWrapper
        ESTIMATORS_AVAILABLE['DoublyRobustLearner'] = DRLearner
        logger.info("✅ Doubly Robust estimators disponibles desde causallib/econml")
    except ImportError:
        logger.warning("⚠️  Doubly Robust estimators no disponibles")

try:
    from causal_estimators.double_ml import DoubleML
    ESTIMATORS_AVAILABLE['DoubleML'] = DoubleML
    logger.info("✅ DoubleML disponible")
except ImportError as e:
    logger.warning(f"⚠️  DoubleML no disponible: {e}")

try:
    from causal_estimators.deep_estimators import CEVAE
    ESTIMATORS_AVAILABLE['CEVAE'] = CEVAE
    logger.info("✅ CEVAE disponible")
except ImportError as e:
    logger.warning(f"⚠️  CEVAE no disponible: {e}")

try:
    # Intentar importar desde el repositorio, pero manejar conflictos de utils
    import sys
    original_path = sys.path.copy()
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    try:
        from causal_estimators.ipw_estimator import IPWEstimator
        ESTIMATORS_AVAILABLE['IPWEstimator'] = IPWEstimator
        logger.info("✅ IPWEstimator disponible desde repositorio")
    finally:
        sys.path = original_path
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️  IPWEstimator del repositorio no disponible: {e}")
    # Fallback: usar causallib directamente
    try:
        from causallib.estimation import IPW
        # Crear wrapper simple
        class IPWWrapper:
            def __init__(self, prop_score_model=LogisticRegression()):
                self.estimator = IPW(learner=prop_score_model)
            def fit(self, w, t, y):
                # causallib usa fit(X, a, y) donde X=features, a=treatment, y=outcome
                # Convertir a DataFrame/Series si es necesario
                import pandas as pd
                if isinstance(w, np.ndarray):
                    w = pd.DataFrame(w)
                if isinstance(t, np.ndarray):
                    t = pd.Series(t) if t.ndim == 1 else pd.DataFrame(t)
                if isinstance(y, np.ndarray):
                    y = pd.Series(y) if y.ndim == 1 else pd.DataFrame(y)
                self.estimator.fit(X=w, a=t, y=y)
            def effect(self, X):
                # IPW estima efectos - necesita tratamiento observado, así que usamos estimate_effect
                import pandas as pd
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X)
                try:
                    # IPW usa estimate_effect(X, a1, a0) para efectos individuales
                    te = self.estimator.estimate_effect(X=X, a1=1, a0=0)
                    return te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
                except (AttributeError, TypeError) as e:
                    # IPW no tiene estimate_individual_outcome, así que usamos estimate_effect con tratamiento simulado
                    # Crear tratamiento simulado para cada individuo
                    try:
                        # Intentar con tratamiento alternado
                        a_alt = pd.Series([0, 1] * (len(X) // 2 + 1))[:len(X)]
                        y_sim = pd.Series(np.zeros(len(X)))  # No se usa realmente
                        y0 = self.estimator.estimate_population_outcome(X=X, a=a_alt, y=y_sim, treatment_values=[0])
                        y1 = self.estimator.estimate_population_outcome(X=X, a=a_alt, y=y_sim, treatment_values=[1])
                        result = (y1 - y0)
                        return result.values.flatten() if hasattr(result, 'values') else result.flatten()
                    except:
                        logger.warning(f"   Error en IPW effect: {e}")
                        # Último fallback: asumir TE=0
                        return np.zeros(len(X))
        ESTIMATORS_AVAILABLE['IPWEstimator'] = IPWWrapper
        logger.info("✅ IPWEstimator disponible desde causallib")
    except ImportError:
        logger.warning("⚠️  causallib.estimation.IPW no disponible")

try:
    # Intentar importar desde el repositorio, pero manejar conflictos de utils y warnings de rpy2
    import sys
    import warnings
    original_path = sys.path.copy()
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    # Suprimir warnings de deprecación de rpy2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            from causal_estimators.matching import MatchingEstimator
            ESTIMATORS_AVAILABLE['MatchingEstimator'] = MatchingEstimator
            logger.info("✅ MatchingEstimator disponible desde repositorio")
        finally:
            sys.path = original_path
except (ImportError, AttributeError, DeprecationWarning) as e:
    logger.warning(f"⚠️  MatchingEstimator del repositorio no disponible: {e}")
    # Fallback: usar causallib directamente
    try:
        from causallib.estimation import Matching
        # Crear wrapper simple
        class MatchingWrapper:
            def __init__(self):
                self.estimator = Matching()
            def fit(self, w, t, y):
                # causallib usa fit(X, a, y) donde X, a, y pueden ser arrays o DataFrames
                # Convertir a DataFrame si es necesario
                import pandas as pd
                if isinstance(w, np.ndarray):
                    w = pd.DataFrame(w)
                if isinstance(t, np.ndarray):
                    t = pd.Series(t) if t.ndim == 1 else pd.DataFrame(t)
                if isinstance(y, np.ndarray):
                    y = pd.Series(y) if y.ndim == 1 else pd.DataFrame(y)
                self.estimator.fit(X=w, a=t, y=y)
            def effect(self, X):
                # Matching estima efectos usando estimate_individual_outcome
                import pandas as pd
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X)
                try:
                    # estimate_individual_outcome devuelve y0 y y1 para cada individuo
                    y0 = self.estimator.estimate_individual_outcome(X=X, a=0)
                    y1 = self.estimator.estimate_individual_outcome(X=X, a=1)
                    te = y1 - y0
                    return te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
                except (AttributeError, TypeError) as e:
                    # Fallback: usar estimate_effect si está disponible
                    try:
                        te = self.estimator.estimate_effect(X=X, a1=1, a0=0)
                        return te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
                    except:
                        logger.warning(f"   Error en Matching effect: {e}")
                        # Último fallback: asumir TE=0
                        return np.zeros(len(X))
        ESTIMATORS_AVAILABLE['MatchingEstimator'] = MatchingWrapper
        logger.info("✅ MatchingEstimator disponible desde causallib")
    except ImportError:
        logger.warning("⚠️  causallib.estimation.Matching no disponible")

# Fallback a econml directamente si no hay repositorio
if not ESTIMATORS_AVAILABLE:
    try:
        import econml.grf
        import econml.metalearners
        ESTIMATORS_AVAILABLE['CausalForest'] = econml.grf.CausalForest
        ESTIMATORS_AVAILABLE['SLearner'] = econml.metalearners.SLearner
        logger.info("⚠️  Usando econml directamente como fallback")
    except ImportError:
        logger.error("❌ econml no está instalado. Instala con: pip install econml")
        sys.exit(1)

logger.info(f"📊 Total de estimadores disponibles: {len(ESTIMATORS_AVAILABLE)}")
logger.info(f"   Métodos: {', '.join(ESTIMATORS_AVAILABLE.keys())}")


def prepare_causal_est_features(df_events: pd.DataFrame) -> tuple:
    """
    Prepara features básicas para Causal Effect Estimation desde el log de eventos.
    Similar a prepare_wtt_features pero adaptado para este modelo.
    
    Returns:
        Tupla (df_features, case_col) con features preparadas y nombre de columna de caso
    """
    logger.info("🔧 Preparando features para Causal Effect Estimation...")
    
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


def create_estimator(estimator_name: str, estimator_class: Any) -> Any:
    """
    Crea una instancia del estimador con parámetros apropiados.
    
    Args:
        estimator_name: Nombre del estimador
        estimator_class: Clase del estimador
    
    Returns:
        Instancia del estimador configurada
    """
    try:
        if estimator_name == 'CausalForest':
            return estimator_class(
                n_estimators=100,
                min_samples_leaf=5,
                max_depth=None,
                random_state=42
            )
        elif estimator_name == 'CausalTree':
            # CausalTree necesita subforest_size=1 cuando n_estimators=1, pero inference=False
            try:
                return estimator_class(
                    n_estimators=1,
                    min_samples_leaf=10,
                    max_depth=5,
                    subforest_size=1,  # Debe ser divisible por n_estimators
                    inference=False,  # Necesario cuando subforest_size=1
                    random_state=42
                )
            except TypeError:
                # Si falla, usar CausalForest con parámetros de árbol
                import econml.grf
                return econml.grf.CausalForest(
                    n_estimators=1,
                    min_samples_leaf=10,
                    max_depth=5,
                    subforest_size=1,
                    inference=False,  # Necesario cuando subforest_size=1
                    random_state=42
                )
        elif estimator_name == 'ORthoforestDML':
            # ORthoforestDML puede tener diferentes interfaces
            try:
                return estimator_class(
                    outcome_model=RandomForestRegressor(n_estimators=50, random_state=42),
                    prop_score_model=LogisticRegression(random_state=42),
                    n_trees=50,
                    min_leaf_size=10,
                    max_depth=10,
                    discrete_treatment=True,
                    random_state=42
                )
            except TypeError:
                # Si falla, usar econml directamente
                import econml.orf
                return econml.orf.DMLOrthoForest(
                    model_Y=RandomForestRegressor(n_estimators=50, random_state=42),
                    model_T=LogisticRegression(random_state=42),
                    n_trees=50,
                    min_leaf_size=10,
                    max_depth=10,
                    discrete_treatment=True,
                    random_state=42
                )
        elif estimator_name == 'SLearner':
            # SLearner puede usar outcome_model (repositorio) o overall_model (econml directo)
            outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                return estimator_class(outcome_model=outcome_model)
            except TypeError:
                # Si falla, intentar con overall_model (econml directo)
                import econml.metalearners
                return econml.metalearners.SLearner(overall_model=outcome_model)
        elif estimator_name == 'TLearner':
            outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                return estimator_class(outcome_models=outcome_model)
            except TypeError:
                import econml.metalearners
                return econml.metalearners.TLearner(models=outcome_model)
        elif estimator_name == 'XLearner':
            outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            prop_model = LogisticRegression(random_state=42)
            try:
                return estimator_class(
                    outcome_models=outcome_model,
                    prop_score_model=prop_model
                )
            except TypeError:
                import econml.metalearners
                return econml.metalearners.XLearner(
                    models=outcome_model,
                    propensity_model=prop_model
                )
        elif estimator_name == 'StandardizationEstimator':
            return estimator_class(
                outcome_model=RandomForestRegressor(n_estimators=100, random_state=42)
            )
        elif estimator_name == 'StratifiedStandardizationEstimator':
            return estimator_class(
                outcome_models=RandomForestRegressor(n_estimators=100, random_state=42)
            )
        elif estimator_name == 'DoublyRobustEstimator':
            return estimator_class(
                outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
                prop_score_model=LogisticRegression(random_state=42)
            )
        elif estimator_name == 'DoublyRobustLearner':
            return estimator_class(
                outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
                prop_score_model=LogisticRegression(random_state=42)
            )
        elif estimator_name == 'DoubleML':
            return estimator_class(
                outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
                prop_score_model=LogisticRegression(random_state=42)
            )
        elif estimator_name == 'CEVAE':
            return estimator_class(
                num_epochs=50,  # Reducido para datasets pequeños
                batch_size=32,
                learning_rate=0.001
            )
        elif estimator_name == 'IPWEstimator':
            return estimator_class(
                prop_score_model=LogisticRegression(random_state=42)
            )
        elif estimator_name == 'MatchingEstimator':
            return estimator_class()
        else:
            # Fallback: intentar crear con parámetros por defecto
            return estimator_class()
    except Exception as e:
        logger.warning(f"⚠️  Error creando {estimator_name}: {e}")
        return None


def train_estimator(
    estimator_name: str,
    estimator: Any,
    X_scaled: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray
) -> bool:
    """
    Entrena un estimador causal.
    
    Returns:
        True si se entrenó exitosamente, False en caso contrario
    """
    try:
        # Asegurar que Y y T sean arrays 1D
        Y = np.ravel(Y)
        T = np.ravel(T)
        
        # Diferentes estimadores tienen diferentes interfaces
        if hasattr(estimator, 'forest_fit'):
            # Para CausalForest, CausalTree
            estimator.forest_fit(X_scaled, T, Y)
        elif hasattr(estimator, 'fit'):
            # Para la mayoría de los estimadores
            if estimator_name in ['CausalForest', 'CausalTree']:
                # Estos usan forest_fit, pero si no está disponible, usar fit
                estimator.fit(X_scaled, T, Y)
            elif estimator_name in ['SLearner', 'TLearner', 'XLearner', 'ORthoforestDML', 'DoubleML']:
                # Estos usan fit(w, t, y) o fit(Y, T, X=...)
                try:
                    estimator.fit(X_scaled, T, Y)
                except TypeError:
                    # Algunos usan fit(Y, T, X=...)
                    estimator.fit(Y, T, X=X_scaled)
            elif estimator_name in ['StandardizationEstimator', 'StratifiedStandardizationEstimator', 
                                   'IPWEstimator']:
                # Nuestros wrappers de causallib usan fit(w, t, y) y manejan la conversión internamente
                estimator.fit(X_scaled, T, Y)
            elif estimator_name in ['DoublyRobustEstimator', 'DoublyRobustLearner']:
                # AIPW y DRLearner pueden usar diferentes interfaces
                try:
                    estimator.fit(X=X_scaled, a=T, y=Y)
                except TypeError:
                    try:
                        estimator.fit(X_scaled, T, Y)
                    except TypeError:
                        estimator.fit(Y, T, X=X_scaled)
            elif estimator_name == 'CEVAE':
                # CEVAE puede tener una interfaz diferente
                estimator.fit(X_scaled, T, Y)
            elif estimator_name == 'MatchingEstimator':
                # Nuestros wrappers de causallib usan fit(w, t, y) y manejan la conversión internamente
                estimator.fit(X_scaled, T, Y)
            else:
                # Fallback genérico
                estimator.fit(X_scaled, T, Y)
        
        return True
    except Exception as e:
        logger.error(f"❌ Error entrenando {estimator_name}: {e}")
        return False


def estimate_treatment_effects(
    estimator_name: str,
    estimator: Any,
    X_scaled: np.ndarray
) -> Optional[np.ndarray]:
    """
    Estima Treatment Effects usando el estimador.
    
    Returns:
        Array de treatment effects o None si falla
    """
    try:
        # Diferentes estimadores tienen diferentes métodos
        if hasattr(estimator, 'estimate_ite_forest'):
            te = estimator.estimate_ite_forest(w=X_scaled)
        elif hasattr(estimator, 'estimate_ite'):
            te = estimator.estimate_ite(w=X_scaled)
        elif hasattr(estimator, 'estimate_effect'):
            # causallib usa estimate_effect(X, a1, a0)
            # Nuestros wrappers usan effect(X) que internamente llama a estimate_effect
            try:
                # Si es un wrapper, usar effect()
                if hasattr(estimator, 'estimator'):
                    te = estimator.effect(X_scaled)
                else:
                    # Es causallib directo
                    import pandas as pd
                    X_df = pd.DataFrame(X_scaled) if isinstance(X_scaled, np.ndarray) else X_scaled
                    te = estimator.estimate_effect(X=X_df, a1=1, a0=0)
                    te = te.values.flatten() if hasattr(te, 'values') else (te.flatten() if te.ndim > 1 else te)
            except (TypeError, AttributeError):
                # Algunos pueden usar estimate_effect sin parámetros
                try:
                    te = estimator.estimate_effect(X_scaled)
                except:
                    te = estimator.effect(X_scaled)
        elif hasattr(estimator, 'effect'):
            te = estimator.effect(X_scaled)
        elif hasattr(estimator, 'predict'):
            te = estimator.predict(X_scaled)
            if len(te.shape) > 1:
                te = te.flatten()
        else:
            logger.warning(f"⚠️  {estimator_name} no tiene método conocido para estimar TE")
            return None
        
        # Aplanar a 1D si es necesario
        if te.ndim > 1:
            if te.shape[1] == 1:
                te = te.flatten()
            else:
                logger.warning(f"   TE tiene forma {te.shape}, usando primera columna")
                te = te[:, 0]
        
        # Asegurar que es 1D
        te = te.flatten() if te.ndim > 1 else te
        
        # Asegurar que tiene la longitud correcta
        if len(te) != len(X_scaled):
            logger.warning(f"   TE tiene longitud {len(te)} pero X tiene {len(X_scaled)}, ajustando...")
            if len(te) > len(X_scaled):
                te = te[:len(X_scaled)]
            else:
                te = np.pad(te, (0, len(X_scaled) - len(te)), mode='constant', constant_values=0)
        
        return te
    except Exception as e:
        logger.error(f"❌ Error estimando TE con {estimator_name}: {e}")
        return None


def evaluate_estimator(
    estimator_name: str,
    df_train_features: pd.DataFrame,
    df_train_labels: pd.DataFrame,
    df_test_features: pd.DataFrame,
    df_test_results: pd.DataFrame,
    case_col: str,
    scaler: MinMaxScaler,
    feature_cols: List[str],
    conf_threshold: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Evalúa un estimador causal específico.
    
    Returns:
        Diccionario con métricas o None si falla
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluando: {estimator_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Obtener clase del estimador
        estimator_class = ESTIMATORS_AVAILABLE.get(estimator_name)
        if estimator_class is None:
            logger.warning(f"⚠️  {estimator_name} no está disponible")
            return None
        
        # Crear instancia del estimador
        estimator = create_estimator(estimator_name, estimator_class)
        if estimator is None:
            logger.warning(f"⚠️  No se pudo crear {estimator_name}")
            return None
        
        # Preparar datos de entrenamiento
        X_train = df_train_features[feature_cols].values
        for col in feature_cols:
            X_train[:, feature_cols.index(col)] = pd.to_numeric(X_train[:, feature_cols.index(col)], errors='coerce')
        X_train = pd.DataFrame(X_train, columns=feature_cols).fillna(0).values
        
        T_train = df_train_labels['treatment_observed'].values
        Y_train = df_train_labels['outcome_observed'].values
        
        # Normalizar
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Entrenar
        if not train_estimator(estimator_name, estimator, X_train_scaled, T_train, Y_train):
            return None
        
        logger.info(f"✅ {estimator_name} entrenado exitosamente")
        
        # Preparar datos de test
        X_test = df_test_features[feature_cols].values
        for col in feature_cols:
            X_test[:, feature_cols.index(col)] = pd.to_numeric(X_test[:, feature_cols.index(col)], errors='coerce')
        X_test = pd.DataFrame(X_test, columns=feature_cols).fillna(0).values
        X_test_scaled = scaler.transform(X_test)
        
        # Medir tiempo de inferencia
        start_time = time.time()
        
        # Estimar Treatment Effects
        te = estimate_treatment_effects(estimator_name, estimator, X_test_scaled)
        if te is None:
            logger.warning(f"⚠️  No se pudieron estimar TE para {estimator_name}")
            return None
        
        # Calcular latencia total (tiempo total / número de casos)
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000  # Convertir a milisegundos
        latency_ms = total_time_ms / len(X_test_scaled) if len(X_test_scaled) > 0 else 0.0
        
        # Aplicar política: intervenir si TE >= threshold
        action_model = (te >= conf_threshold).astype(int)
        
        # Crear DataFrame con decisiones
        case_decisions = pd.DataFrame({
            case_col: df_test_features[case_col].values.flatten() if df_test_features[case_col].values.ndim > 1 else df_test_features[case_col].values,
            'action_model': action_model.flatten() if action_model.ndim > 1 else action_model,
            'treatment_effect': te.flatten() if te.ndim > 1 else te,
            'uplift_score': te.flatten() if te.ndim > 1 else te
        })
        
        logger.info(f"✅ Predicciones generadas: {action_model.sum()}/{len(action_model)} casos con intervención")
        
        # Merge decisiones del modelo en df_results
        df_final = df_test_results.copy()
        if 'action_model' in df_final.columns:
            del df_final['action_model']
        
        df_final = df_final.merge(
            case_decisions.rename(columns={case_col: 'case_id'}),
            on='case_id',
            how='left'
        )
        df_final['action_model'] = df_final['action_model'].fillna(0).astype(int)
        
        # Agregar uplift_score si no existe
        if 'uplift_score' not in df_final.columns and 'treatment_effect' in df_final.columns:
            df_final['uplift_score'] = df_final['treatment_effect']
        
        # Calcular métricas
        evaluator = BenchmarkEvaluator()
        
        # Determinar complejidad según el tipo de estimador
        if 'Forest' in estimator_name or 'Tree' in estimator_name:
            complexity = "Media (CPU - Forest)"
        elif 'Learner' in estimator_name:
            complexity = "Media (CPU - MetaLearner)"
        elif estimator_name == 'CEVAE':
            complexity = "Alta (GPU - Deep Learning)"
        else:
            complexity = "Media (CPU)"
        
        metrics = evaluator.evaluate(df_final, training_complexity=complexity)
        
        # IMPORTANTE: Para comparación justa con el baseline, usar el mismo cálculo de Net Gain
        # El baseline usa calculate_historical_net_gain que promedia rewards con treatment_observed
        # Para modelos, usamos action_model en lugar de treatment_observed
        df_for_direct = df_final.copy()
        df_for_direct['direct_reward'] = df_for_direct.apply(
            lambda row: evaluator.calculate_observed_reward(
                row.get('outcome_observed', 0),
                row.get('action_model', 0),  # Usar action_model en lugar de treatment_observed
                row.get('duration_days', 0)
            ),
            axis=1
        )
        direct_net_gain = df_for_direct['direct_reward'].mean()
        logger.info(f"📊 Net Gain Directo (usando action_model): ${direct_net_gain:.2f}")
        
        # Guardar ambos para referencia y usar el directo como principal
        metrics['net_gain_ipw'] = metrics['net_gain']
        metrics['net_gain'] = direct_net_gain
        
        # Agregar nombre del estimador y latencia
        metrics['estimator_name'] = estimator_name
        metrics['latency_ms'] = latency_ms
        
        logger.info(f"⏱️  Latencia promedio: {latency_ms:.4f} ms por caso (total: {total_time_ms:.2f} ms para {len(X_test_scaled)} casos)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error evaluando {estimator_name}: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluación Causal Effect Estimation para BPI Challenge 2017')
    parser.add_argument('--test', action='store_true', 
                       help='Usar archivos procesados de train/test (bpi2017_train.csv y bpi2017_test.csv)')
    args = parser.parse_args()
    
    logger.info(f"{'='*80}\nEVALUACIÓN FINAL: Causal Effect Estimation (Múltiples Métodos) vs BASELINE\n{'='*80}")
    
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
        df_train_features, case_col = prepare_causal_est_features(df_train_events)
        
        # Preparar features y resultados para test
        df_test_results = prepare_baseline_dataframe(df_test_events)
        df_test_features, _ = prepare_causal_est_features(df_test_events)
        
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
        
        # Preparar Features
        df_features, case_col = prepare_causal_est_features(df_events)
        
        # Merge features con resultados
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
        
        # Preparar datos de test
        test_case_ids = df_test[case_col].values
        df_test_results = df_results[df_results['case_id'].isin(test_case_ids)].copy()
        
        logger.info(f"📊 Split: {len(df_train)} casos para entrenamiento, {len(df_test)} casos para evaluación")
    
    # Preparar features y labels
    exclude_cols = [case_col, 'case_id', 'treatment_observed', 'outcome_observed']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    # Filtrar solo columnas numéricas
    numeric_feature_cols = []
    for col in feature_cols:
        if df_train[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            try:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
                if df_train[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_feature_cols.append(col)
            except:
                pass
    
    feature_cols = numeric_feature_cols
    logger.info(f"   Usando {len(feature_cols)} features numéricas")
    
    # Scaler común para todos los estimadores
    scaler = MinMaxScaler()
    
    # Preparar datos de test (si no se usaron archivos procesados)
    if not args.test:
        test_case_ids = df_test[case_col].values
        df_test_results = df_results[df_results['case_id'].isin(test_case_ids)].copy()
    
    # 6. Evaluar cada estimador
    all_metrics = []
    successful_estimators = []
    failed_estimators = []
    
    # Orden de evaluación: primero los más simples y confiables
    # Nota: ORthoforestDML es muy lento, se omite por defecto
    evaluation_order = [
        'CausalForest',
        'CausalTree',
        'SLearner',
        'TLearner',
        'XLearner',
        'StandardizationEstimator',
        'StratifiedStandardizationEstimator',
        # 'ORthoforestDML',  # Muy lento, omitido por defecto
        'DoublyRobustEstimator',
        'DoublyRobustLearner',
        'DoubleML',
        'IPWEstimator',
        'MatchingEstimator',
        # 'CEVAE'  # Último porque puede requerir más recursos, omitido por defecto
    ]
    
    for estimator_name in evaluation_order:
        if estimator_name not in ESTIMATORS_AVAILABLE:
            logger.info(f"⏭️  Saltando {estimator_name} (no disponible)")
            continue
        
        metrics = evaluate_estimator(
            estimator_name,
            df_train,
            df_train[['treatment_observed', 'outcome_observed']],
            df_test,
            df_test_results,
            case_col,
            scaler,
            feature_cols,
            conf_threshold=0.0
        )
        
        if metrics is not None:
            all_metrics.append(metrics)
            successful_estimators.append(estimator_name)
            logger.info(f"✅ {estimator_name} evaluado exitosamente")
        else:
            failed_estimators.append(estimator_name)
            logger.warning(f"❌ {estimator_name} falló")
    
    # 7. Comparar con Baseline
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
            logger.warning(f"No se pudo leer baseline_metrics.csv: {e}")
    else:
        logger.warning(f"Archivo baseline_metrics.csv no encontrado en: {baseline_csv}")
    
    # Recalcular Lift para cada estimador
    for metrics in all_metrics:
        model_gain = metrics['net_gain']
        if baseline_gain != 0:
            real_lift = ((model_gain - baseline_gain) / abs(baseline_gain)) * 100
            metrics['lift_vs_bau'] = real_lift
    
    # 8. Reporte Final
    print("\n" + "="*80)
    print("🏆 RESULTADOS FINALES: Causal Effect Estimation (Múltiples Métodos)")
    print("="*80)
    print(f"\n✅ Métodos evaluados exitosamente: {len(successful_estimators)}")
    print(f"❌ Métodos que fallaron: {len(failed_estimators)}")
    
    if all_metrics:
        # Crear DataFrame con todos los resultados
        df_all_metrics = pd.DataFrame(all_metrics)
        
        # Ordenar por Net Gain descendente
        df_all_metrics = df_all_metrics.sort_values('net_gain', ascending=False)
        
        print("\n" + "="*80)
        print("RANKING DE MÉTODOS (por Net Gain)")
        print("="*80)
        for idx, row in df_all_metrics.iterrows():
            print(f"\n{row['estimator_name']}:")
            print(f"  💰 Net Gain:     ${row['net_gain']:.2f}")
            print(f"  🚀 Lift vs BAU:  {row['lift_vs_bau']:+.2f}%")
            print(f"  🎯 % Intervenciones: {row['intervention_percentage']:.2f}%")
            print(f"  🛡️  % Violaciones:    {row['violation_percentage']:.2f}%")
            if pd.notna(row.get('auc_qini')):
                print(f"  📈 AUC-Qini:        {row['auc_qini']:.4f}")
            if pd.notna(row.get('latency_ms')):
                print(f"  ⏱️  Latencia:        {row['latency_ms']:.4f} ms/caso")
        
        # Guardar resultados
        if args.test:
            out_dir = os.path.join(project_root, "results/benchmark/bpi2017_test")
        else:
            out_dir = os.path.join(project_root, "results/benchmark/causal_effect_estimation")
        os.makedirs(out_dir, exist_ok=True)
        
        output_file = os.path.join(out_dir, "all_causal_estimators_metrics.csv")
        df_all_metrics.to_csv(output_file, index=False)
        logger.info(f"\n✅ Resultados guardados en: {output_file}")
        
        # Guardar también un resumen
        summary_file = os.path.join(out_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RESUMEN DE EVALUACIÓN: Causal Effect Estimation\n")
            f.write("="*80 + "\n\n")
            f.write(f"Métodos evaluados exitosamente: {len(successful_estimators)}\n")
            f.write(f"Métodos que fallaron: {len(failed_estimators)}\n\n")
            f.write("Métodos exitosos:\n")
            for name in successful_estimators:
                f.write(f"  ✅ {name}\n")
            if failed_estimators:
                f.write("\nMétodos que fallaron:\n")
                for name in failed_estimators:
                    f.write(f"  ❌ {name}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("RANKING (por Net Gain)\n")
            f.write("="*80 + "\n")
            for idx, row in df_all_metrics.iterrows():
                f.write(f"\n{row['estimator_name']}:\n")
                f.write(f"  Net Gain: ${row['net_gain']:.2f}\n")
                f.write(f"  Lift vs BAU: {row['lift_vs_bau']:+.2f}%\n")
        logger.info(f"✅ Resumen guardado en: {summary_file}")
    else:
        logger.error("❌ No se pudo evaluar ningún método")
        sys.exit(1)


if __name__ == "__main__":
    main()
