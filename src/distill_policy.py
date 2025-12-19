#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3: POLICY DISTILLATION (Imitation Learning)
-------------------------------------------------
Este script toma el 'Experience Buffer' generado por el Causal-Gym (Fase 2)
y entrena un modelo interpretable y ultrarrápido (Student) para producción.

Objetivos:
1. Filtrar comportamientos inseguros o de baja recompensa.
2. Entrenar un Decision Tree (White-Box) que imite al Agente RL.
3. Exportar reglas SQL/IF-THEN para auditoría.
4. Benchmarking de latencia para demostrar superioridad (<1ms).
"""

import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_config(config_path=None):
    """Carga la configuración desde el archivo YAML"""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == "src":
            base_dir = os.path.dirname(script_dir)
        else:
            base_dir = script_dir
        config_path = os.path.join(base_dir, "configs/config.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ No se encontró archivo de configuración: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error leyendo configuración: {e}")
        return None

def load_experience_buffer(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: No se encontró el buffer en {file_path}")
        print("   Ejecuta primero la Fase 2 (train_agent_in_gym.py)")
        sys.exit(1)
    
    print(f"📂 Cargando Experience Buffer: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Total de experiencias crudas: {len(df)}")
    return df

def filter_high_quality_experiences(df, quality_threshold=0.0):
    """
    Estrategia de Distilación:
    Solo aprendemos de las acciones que fueron:
    1. SEGURAS (was_safe == True) -> Garantiza Compliance 100%
    2. EXITOSAS (reward_causal > umbral) -> Garantiza Profit
    """
    print("\n🧹 Filtrando experiencias para el 'Student Model'...")
    
    # 1. Filtro de Seguridad
    initial_len = len(df)
    df_safe = df[df['was_safe'] == True].copy()
    print(f"   - Eliminadas {initial_len - len(df_safe)} acciones inseguras (Violaciones LTL).")
    
    # 2. Filtro de Calidad (Profit)
    if 'reward_causal' in df_safe.columns:
        if quality_threshold is None or quality_threshold == 0.0:
            # Si no se especifica umbral, usar el median como fallback
            threshold = df_safe['reward_causal'].median()
            print(f"   - Usando umbral automático (mediana): {threshold:.2f}")
        else:
            threshold = quality_threshold
            print(f"   - Usando umbral configurado: {threshold:.2f}")
        
        df_elite = df_safe[df_safe['reward_causal'] >= threshold].copy()
        print(f"   - Filtrando acciones sub-óptimas (Reward < {threshold:.2f})")
    else:
        df_elite = df_safe
        print("   - No se encontró columna 'reward_causal', usando todas las experiencias seguras")
        
    print(f"   Dataset final de entrenamiento: {len(df_elite)} ejemplos de alta calidad.")
    return df_elite

def train_student_model(df, max_depth=5, criterion='entropy', test_size=0.2, random_state=42):
    """
    Entrena un Árbol de Decisión simple para imitar al Agente RL.
    """
    print("\n🧠 Entrenando 'Student Model' (Decision Tree)...")
    print(f"   • Profundidad máxima: {max_depth}")
    print(f"   • Criterio: {criterion}")
    print(f"   • Tamaño de prueba: {test_size*100:.0f}%")
    
    # Features y Target
    # En train_agent_in_gym.py, el estado era un string "gateway=...".
    # Usamos CountVectorizer para convertir eso a vector numérico simple.
    X_raw = df['state_feature_vector']
    y = df['action_taken']
    
    # Pipeline: Vectorización -> Árbol
    # max_depth asegura que el modelo sea interpretable por humanos (White-Box)
    model = Pipeline([
        ('vectorizer', CountVectorizer(binary=True)),
        ('classifier', DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=test_size, random_state=random_state)
    
    model.fit(X_train, y_train)
    
    # Evaluación
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Student Model Accuracy (Imitación del Maestro): {acc*100:.2f}%")
    
    return model

def generate_white_box_rules(model):
    """
    Extrae las reglas IF-THEN del árbol para demostrar explicabilidad.
    """
    print("\n📜 REGLAS DE NEGOCIO GENERADAS (White-Box Policy):")
    print("-" * 60)
    
    tree = model.named_steps['classifier']
    vec = model.named_steps['vectorizer']
    feature_names = vec.get_feature_names_out()
    
    rules_text = export_text(tree, feature_names=list(feature_names))
    print(rules_text)
    print("-" * 60)
    print("💡 Estas reglas pueden exportarse directamente a SQL o Java Drools.")

def benchmark_latency(model, sample_input):
    """
    Prueba de fuego: Latencia de inferencia.
    Demuestra por qué esto gana a las Redes Neuronales y Conformal Prediction.
    """
    print("\n🏎️  BENCHMARK DE LATENCIA (Producción):")
    
    iterations = 10000
    start_time = time.time()
    
    # Simulamos batch size = 1 (Tiempo Real puro)
    for _ in range(iterations):
        _ = model.predict([sample_input])
        
    total_time = time.time() - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    
    print(f"   Inferencia promedio (CPU): {avg_latency_ms:.4f} ms")
    
    if avg_latency_ms < 1.0:
        print("   🚀 RESULTADO: LATENCIA < 1ms (Gana el Benchmark)")
    else:
        print("   ⚠️  Latencia alta, revisar profundidad del árbol.")

def main():
    # Cargar configuración
    config = load_config()
    if config is None:
        print("❌ No se pudo cargar la configuración desde configs/config.yaml")
        sys.exit(1)
    
    distill_config = config.get("distill_config", {})
    script_config = config.get("script_config", {})
    
    # Obtener directorio base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    # Obtener rutas de entrada y salida
    input_csv = distill_config.get("input_csv")
    if input_csv is None:
        # Usar ruta por defecto
        rl_output_dir = script_config.get("rl_output_dir")
        if rl_output_dir is None:
            rl_output_dir = os.path.join(base_dir, "data", "generado-rl-train")
        else:
            # Si es relativa, hacerla absoluta
            if not os.path.isabs(rl_output_dir):
                rl_output_dir = os.path.join(base_dir, rl_output_dir)
            else:
                rl_output_dir = os.path.abspath(rl_output_dir)
        input_csv = os.path.join(rl_output_dir, "experience_buffer.csv")
    else:
        # Si es relativa, hacerla absoluta
        if not os.path.isabs(input_csv):
            input_csv = os.path.join(base_dir, input_csv)
    
    output_model = distill_config.get("output_model")
    if output_model is None:
        # Usar ruta por defecto
        distill_output_dir = script_config.get("distill_output_dir")
        if distill_output_dir is None:
            distill_output_dir = os.path.join(base_dir, "data")
        else:
            # Si es relativa, hacerla absoluta
            if not os.path.isabs(distill_output_dir):
                distill_output_dir = os.path.join(base_dir, distill_output_dir)
            else:
                distill_output_dir = os.path.abspath(distill_output_dir)
        output_model = os.path.join(distill_output_dir, "final_policy_model.pkl")
    else:
        # Si es relativa, hacerla absoluta
        if not os.path.isabs(output_model):
            output_model = os.path.join(base_dir, output_model)
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    # Obtener parámetros de configuración
    max_depth = distill_config.get("max_depth", 5)
    criterion = distill_config.get("criterion", "entropy")
    test_size = distill_config.get("test_size", 0.2)
    quality_threshold = distill_config.get("quality_threshold", 0.0)
    
    print("=" * 80)
    print("📚 DESTILACIÓN DE POLÍTICA (Policy Distillation)")
    print("=" * 80)
    print(f"📋 Configuración:")
    print(f"   • Archivo de entrada: {input_csv}")
    print(f"   • Archivo de salida: {output_model}")
    print(f"   • Profundidad máxima: {max_depth}")
    print(f"   • Criterio: {criterion}")
    print(f"   • Umbral de calidad: {quality_threshold}")
    print()
    
    # 1. Cargar Datos
    df = load_experience_buffer(input_csv)
    
    if len(df) < 10:
        print("⚠️  Muy pocos datos para destilar. Ejecuta más episodios en Fase 2.")
        return

    # 2. Filtrar (Distillation Strategy)
    df_clean = filter_high_quality_experiences(df, quality_threshold=quality_threshold)
    
    if len(df_clean) == 0:
        print("❌ No quedaron datos después del filtrado (¿Todas las acciones fueron inseguras?).")
        return

    # 3. Entrenar Student
    model = train_student_model(df_clean, max_depth=max_depth, criterion=criterion, test_size=test_size)
    
    # 4. Demostrar Explicabilidad
    generate_white_box_rules(model)
    
    # 5. Demostrar Velocidad
    sample_state = df_clean['state_feature_vector'].iloc[0]
    benchmark_latency(model, sample_state)
    
    # 6. Guardar Modelo Final
    joblib.dump(model, output_model)
    print(f"\n💾 Modelo final guardado en: {output_model}")
    print("   Este archivo .pkl es el que se despliega en producción.")

if __name__ == "__main__":
    main()