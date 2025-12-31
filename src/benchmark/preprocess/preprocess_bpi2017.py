#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Preprocesamiento y División Estratificada (Train/Test) para BPI 2017.
Garantiza la validez científica del benchmark asegurando distribución balanceada
de casos tratados/no tratados y exitosos/fallidos.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_bpi2017(input_path, output_dir):
    print("="*80)
    print("PREPROCESAMIENTO CIENTÍFICO BPI 2017")
    print("="*80)
    print(f"🔄 Cargando dataset completo: {input_path}")
    
    # Cargar datos
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {input_path}")
        return

    # 1. Estandarización de Columnas
    # Ajusta esto si tu CSV tiene otros nombres, basado en tu EDA
    case_col = 'case:concept:name'
    act_col = 'concept:name'
    time_col = 'time:timestamp'
    
    # Asegurar formato de fecha
    print("📅 Procesando fechas...")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, format='mixed')
    
    # 2. Definiciones de Negocio (Basadas en tu EDA)
    print("⚙️  Etiquetando casos (Tratamiento y Outcome)...")
    
    # Tratamiento: Llamadas (Intervención costosa)
    treatment_acts = ['W_Call after offers', 'W_Call incomplete files']
    
    # Éxito: Oferta Aceptada (Tu EDA confirmó que O_Accepted es el éxito)
    success_act = 'O_Accepted'
    
    # 3. Ingeniería de Features para Estratificación
    # Agrupamos por caso para saber si CADA caso tuvo tratamiento y si tuvo éxito
    case_metrics = df.groupby(case_col)[act_col].agg(list).reset_index()
    
    def get_case_labels(activities):
        # Es tratado si contiene ALGUNA actividad de tratamiento
        is_treated = any(act in activities for act in treatment_acts)
        # Es exitoso si contiene la actividad de éxito
        is_success = success_act in activities
        return is_treated, is_success

    # Aplicar lógica
    labels = case_metrics[act_col].apply(get_case_labels)
    case_metrics['is_treated'] = [x[0] for x in labels]
    case_metrics['is_success'] = [x[1] for x in labels]
    
    # Crear "Super Clases" para el split estratificado
    # 0: No Tratado - Fallo (La clase más importante para el contrafactual negativo)
    # 1: No Tratado - Éxito (La clase más importante para el contrafactual positivo)
    # 2: Tratado - Fallo
    # 3: Tratado - Éxito
    case_metrics['stratify_group'] = (case_metrics['is_treated'].astype(int) * 2) + case_metrics['is_success'].astype(int)
    
    print("\n📊 Distribución de Grupos en el Dataset Completo:")
    group_counts = case_metrics['stratify_group'].value_counts().sort_index()
    labels_map = {0: "SinTrat/Fallo", 1: "SinTrat/Éxito", 2: "Trat/Fallo", 3: "Trat/Éxito"}
    
    for g, count in group_counts.items():
        print(f"   Grupo {g} ({labels_map[g]}): {count} casos")
        
    # Verificar si hay suficientes datos en los grupos raros
    min_samples = 2
    if any(group_counts < min_samples):
        print(f"\n⚠️  ADVERTENCIA: Algunos grupos tienen menos de {min_samples} casos.")
        print("   Se relajará la estratificación para usar solo 'is_treated'.")
        stratify_col = case_metrics['is_treated']
    else:
        stratify_col = case_metrics['stratify_group']

    # 4. SPLIT TRAIN (70%) / TEST (30%)
    print("\n✂️  Ejecutando Split Estratificado 70/30...")
    
    train_ids, test_ids = train_test_split(
        case_metrics[case_col], 
        test_size=0.30, 
        random_state=42,
        stratify=stratify_col
    )
    
    # 5. Crear DataFrames Finales
    print("💾 Filtrando y guardando archivos...")
    train_df = df[df[case_col].isin(train_ids)]
    test_df = df[df[case_col].isin(test_ids)]
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "bpi2017_train.csv")
    test_path = os.path.join(output_dir, "bpi2017_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 6. Validación Final (Quality Check)
    test_metrics = case_metrics[case_metrics[case_col].isin(test_ids)]
    n_test_untreated = len(test_metrics[test_metrics['is_treated'] == False])
    n_test_success = len(test_metrics[test_metrics['is_success'] == True])
    
    print("\n" + "="*60)
    print("✅ PREPROCESAMIENTO FINALIZADO CON ÉXITO")
    print("="*60)
    print(f"📂 Salida Train: {train_path}")
    print(f"   - Casos: {len(train_ids)}")
    print(f"📂 Salida Test:  {test_path}")
    print(f"   - Casos: {len(test_ids)}")
    print("-" * 60)
    print("🔍 VALIDACIÓN DEL TEST SET (Para el Benchmark):")
    print(f"   - Total Test: {len(test_ids)}")
    print(f"   - Casos SIN Intervención: {n_test_untreated} (Objetivo: >30)")
    print(f"   - Casos Exitosos: {n_test_success}")
    
    if n_test_untreated < 10:
        print("\n⚠️  PELIGRO: Muy pocos casos sin intervención en Test.")
        print("   El cálculo del Lift podría ser inestable.")
    else:
        print("\n✅ El Test Set es válido para calcular Lift vs Baseline.")

if __name__ == "__main__":
    # Calcular rutas absolutas basadas en la ubicación del script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark/preprocess/
    src_dir = os.path.dirname(os.path.dirname(script_dir))  # src/
    project_root = os.path.dirname(src_dir)  # project root
    
    # Ruta donde tienes el archivo original descomprimido o .gz
    logs_dir = os.path.join(project_root, "logs", "BPI2017")
    RAW_LOG = os.path.join(logs_dir, "bpi-challenge-2017.csv")
    
    if not os.path.exists(RAW_LOG):
        # Intentar con .gz si el csv no existe
        print(f"⚠️  CSV no encontrado, intentando con .gz: {RAW_LOG}")
        RAW_LOG = os.path.join(logs_dir, "bpi-challenge-2017.csv.gz")
        if not os.path.exists(RAW_LOG):
            print(f"❌ Error: No se encontró el archivo en {RAW_LOG}")
            print(f"   Buscando en: {logs_dir}")
            print(f"   Archivos disponibles:")
            if os.path.exists(logs_dir):
                for f in os.listdir(logs_dir):
                    print(f"     - {f}")
            exit(1)
        
    # Carpeta donde quieres los archivos listos para usar
    OUTPUT_DIR = os.path.join(logs_dir, "processed")
    
    print(f"📂 Project root: {project_root}")
    print(f"📂 Input file: {RAW_LOG}")
    print(f"📂 Output dir: {OUTPUT_DIR}")
    print()
    
    preprocess_bpi2017(RAW_LOG, OUTPUT_DIR)