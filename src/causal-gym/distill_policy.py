#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3: POLICY DISTILLATION
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logger_utils import setup_logger

logger = setup_logger(__name__)

def load_experience_buffer(file_path):
    if not os.path.exists(file_path):
        logger.error(f"No buffer: {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)

def filter_high_quality_experiences(df, quality_threshold=0.0):
    logger.info("Filtrando experiencias...")
    # 1. Seguridad
    df_safe = df[df['was_safe'] == True].copy()
    
    # 2. Profit (Reward Causal)
    if 'reward_causal' in df_safe.columns:
        if quality_threshold == 0.0:
            threshold = df_safe['reward_causal'].median()
        else:
            threshold = quality_threshold
        df_elite = df_safe[df_safe['reward_causal'] >= threshold].copy()
    else:
        df_elite = df_safe
        
    logger.info(f"Dataset final: {len(df_elite)} muestras")
    return df_elite

def train_student_model(df, max_depth=5):
    logger.info(f"Entrenando Student Tree (depth={max_depth})...")
    
    X = df['state_feature_vector']
    y = df['action_taken']
    
    model = Pipeline([
        ('vectorizer', CountVectorizer(binary=True)),
        ('classifier', DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Accuracy Student: {acc*100:.2f}%")
    return model

def main():
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración")
        sys.exit(1)
    
    # Obtener configuración
    log_config = config.get("log_config", {})
    script_config = config.get("script_config", {})
    distill_config = config.get("distill_config", {})
    
    # Encontrar directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = src/causal-gym/
    src_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    # Detectar si estamos usando train set: verificar si existe results/bpi2017_train/rl
    train_rl_dir = os.path.join(project_root, "results", "bpi2017_train", "rl")
    train_buffer = os.path.join(train_rl_dir, "experience_buffer.csv")
    
    if os.path.exists(train_buffer):
        # Usar train set
        log_name = "bpi2017_train"
        logger.info(f"🎯 Detectado train set: usando {log_name}")
    else:
        # Usar nombre del log desde config.yaml
        log_path = log_config.get("log_path")
        if log_path:
            if not os.path.isabs(log_path):
                log_path = os.path.join(project_root, log_path)
            log_name = get_log_name_from_path(log_path)
        else:
            log_name = "default"
        logger.info(f"📋 Usando nombre del log desde config: {log_name}")
    
    # Determinar ruta del experience buffer (entrada)
    input_csv = distill_config.get("input_csv")
    if input_csv:
        if not os.path.isabs(input_csv):
            input_csv = os.path.join(project_root, input_csv)
    else:
        # Construir desde rl_output_dir
        rl_output_dir_base = script_config.get("rl_output_dir")
        rl_output_dir = build_output_path(rl_output_dir_base, log_name, "rl", default_base="data")
        input_csv = os.path.join(project_root, rl_output_dir, "experience_buffer.csv")
    
    # Búsqueda dinámica del buffer si la ruta construida no existe
    if not os.path.exists(input_csv):
        logger.warning(f"Buffer no encontrado en: {input_csv}")
        logger.info("Buscando experience_buffer.csv en results/...")
        
        # Priorizar bpi2017_train si existe
        if os.path.exists(train_buffer):
            input_csv = train_buffer
            logger.info(f"✅ Buffer de train encontrado: {input_csv}")
        else:
            # Buscar en todos los directorios, pero priorizar los más recientes
            found_buffers = []
            for root, dirs, files in os.walk(os.path.join(project_root, "results")):
                if "experience_buffer.csv" in files:
                    buffer_path = os.path.join(root, "experience_buffer.csv")
                    mtime = os.path.getmtime(buffer_path)
                    found_buffers.append((mtime, buffer_path))
            
            if found_buffers:
                # Ordenar por fecha de modificación (más reciente primero)
                found_buffers.sort(reverse=True)
                input_csv = found_buffers[0][1]
                logger.info(f"✅ Buffer encontrado (más reciente): {input_csv}")
            else:
                logger.error("No se encontró experience_buffer.csv en ningún lugar")
                sys.exit(1)
    
    logger.info(f"Usando experience buffer: {input_csv}")
    
    df = load_experience_buffer(input_csv)
    df_clean = filter_high_quality_experiences(df, quality_threshold=distill_config.get("quality_threshold", 0.0))
    
    if len(df_clean) == 0:
        logger.error("Sin datos para entrenar.")
        sys.exit(1)

    max_depth = distill_config.get("max_depth", 5)
    model = train_student_model(df_clean, max_depth=max_depth)
    
    # Exportar reglas
    tree = model.named_steps['classifier']
    feats = model.named_steps['vectorizer'].get_feature_names_out()
    logger.info(export_text(tree, feature_names=list(feats)))
    
    # Determinar ruta de salida del modelo
    output_model = distill_config.get("output_model")
    if output_model:
        if not os.path.isabs(output_model):
            output_model = os.path.join(project_root, output_model)
    else:
        # Construir desde distill_output_dir
        distill_output_dir_base = script_config.get("distill_output_dir")
        distill_output_dir = build_output_path(distill_output_dir_base, log_name, "distill", default_base="data")
        output_model = os.path.join(project_root, distill_output_dir, "final_policy_model.pkl")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, output_model)
    logger.info(f"Modelo guardado en: {output_model}")

if __name__ == "__main__":
    main()