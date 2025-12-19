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
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_experience_buffer(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: No se encontró el buffer en {file_path}")
        print("   Ejecuta primero la Fase 2 (train_agent_in_gym.py)")
        sys.exit(1)
    
    print(f"📂 Cargando Experience Buffer: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Total de experiencias crudas: {len(df)}")
    return df

def filter_high_quality_experiences(df):
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
    # En un escenario real, usaríamos el percentil 50 o 75 de recompensas.
    # Aquí asumimos que recompensa > 0 implica una buena decisión relativa al baseline.
    if 'reward_causal' in df_safe.columns:
        # Si todas son negativas (costos), tomamos las "menos malas" (top 50%)
        threshold = df_safe['reward_causal'].median()
        df_elite = df_safe[df_safe['reward_causal'] >= threshold].copy()
        print(f"   - Filtrando acciones sub-óptimas (Reward < {threshold:.2f})")
    else:
        df_elite = df_safe
        
    print(f"   Dataset final de entrenamiento: {len(df_elite)} ejemplos de alta calidad.")
    return df_elite

def train_student_model(df):
    """
    Entrena un Árbol de Decisión simple para imitar al Agente RL.
    """
    print("\n🧠 Entrenando 'Student Model' (Decision Tree)...")
    
    # Features y Target
    # En train_agent_in_gym.py, el estado era un string "gateway=...".
    # Usamos CountVectorizer para convertir eso a vector numérico simple.
    X_raw = df['state_feature_vector']
    y = df['action_taken']
    
    # Pipeline: Vectorización -> Árbol
    # max_depth=5 asegura que el modelo sea interpretable por humanos (White-Box)
    model = Pipeline([
        ('vectorizer', CountVectorizer(binary=True)),
        ('classifier', DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
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
    # Rutas relativas basadas en la estructura del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Si estamos en src/, subir un nivel
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
        
    input_csv = os.path.join(base_dir, "data/generado-rl-train/experience_buffer.csv")
    output_model = os.path.join(base_dir, "data/final_policy_model.pkl")
    
    # 1. Cargar Datos
    df = load_experience_buffer(input_csv)
    
    if len(df) < 10:
        print("⚠️  Muy pocos datos para destilar. Ejecuta más episodios en Fase 2.")
        return

    # 2. Filtrar (Distillation Strategy)
    df_clean = filter_high_quality_experiences(df)
    
    if len(df_clean) == 0:
        print("❌ No quedaron datos después del filtrado (¿Todas las acciones fueron inseguras?).")
        return

    # 3. Entrenar Student
    model = train_student_model(df_clean)
    
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