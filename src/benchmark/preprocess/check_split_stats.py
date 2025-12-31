import pandas as pd
import numpy as np
import os

def calculate_metrics(df, name="Dataset"):
    # Definiciones de Negocio (Mismas que EDA)
    terminating_acts = ['A_Accepted', 'A_Denied', 'A_Cancelled', 'A_Complete', 
                        'O_Accepted', 'O_Refused', 'O_Cancelled', 'O_Returned', 
                        'W_Complete application']
    success_acts = ['O_Accepted']
    failure_acts = ['A_Denied', 'A_Cancelled', 'O_Refused', 'O_Cancelled', 'O_Returned']
    intervention_acts = ['W_Call after offers', 'W_Call incomplete files']
    
    # Columnas
    case_col = 'case:concept:name'
    act_col = 'concept:name'
    time_col = 'time:timestamp'
    
    # Cálculos básicos
    n_cases = df[case_col].nunique()
    n_events = len(df)
    n_activities = df[act_col].nunique()
    n_resources = df['org:resource'].nunique() if 'org:resource' in df.columns else 0
    
    # Agrupaciones
    last_acts = df.groupby(case_col)[act_col].last()
    
    # Terminados vs En Curso
    completed = last_acts.isin(terminating_acts).sum()
    ongoing = n_cases - completed
    
    # Éxito vs Fracaso (Basado en presencia de actividad, no solo último evento)
    # Esto es más preciso para BPI 2017
    case_activities = df.groupby(case_col)[act_col].agg(set)
    
    has_success = case_activities.apply(lambda x: any(a in x for a in success_acts))
    # Fracaso: Tiene actividad de fallo Y NO tiene éxito
    has_failure = case_activities.apply(lambda x: any(a in x for a in failure_acts) and not any(a in x for a in success_acts))
    
    # Solo contamos éxito/fracaso sobre los terminados para consistencia con tu tabla
    # (Aunque técnicamente se podría medir en todos)
    completed_ids = last_acts[last_acts.isin(terminating_acts)].index
    n_success = has_success[completed_ids].sum()
    n_failed = has_failure[completed_ids].sum()
    
    # Intervenciones
    has_intervention = case_activities.apply(lambda x: any(a in x for a in intervention_acts))
    n_treated = has_intervention.sum()
    n_untreated = n_cases - n_treated
    
    # Tiempos (usar format='mixed' para manejar diferentes formatos de timestamp)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, format='mixed', errors='coerce')
    case_times = df.groupby(case_col)[time_col].agg(['min', 'max'])
    durations = (case_times['max'] - case_times['min']).dt.total_seconds() / 86400  # días
    
    # Calcular duración del log (del primer al último evento)
    log_start = df[time_col].min()
    log_end = df[time_col].max()
    log_duration_days = (log_end - log_start).days if pd.notna(log_start) and pd.notna(log_end) else 0
    
    # Calcular eventos por caso para mediana
    events_per_case = df.groupby(case_col).size()
    
    # Crear diccionario con todas las métricas (formato completo como EDA)
    metrics = {
        'Dataset': name,
        'Número de casos': n_cases,
        'Número de eventos': n_events,
        'Actividades únicas': n_activities,
        'Recursos únicos': n_resources,
        'Casos terminados': completed,
        'Casos terminados (%)': f"{completed/n_cases:.2%}",
        'Casos en curso': ongoing,
        'Casos en curso (%)': f"{ongoing/n_cases:.2%}",
        'Casos exitosos (de terminados)': n_success if completed > 0 else 0,
        'Casos exitosos (%)': f"{n_success/completed:.2%}" if completed > 0 else "N/A",
        'Casos fracaso (de terminados)': n_failed if completed > 0 else 0,
        'Casos fracaso (%)': f"{n_failed/completed:.2%}" if completed > 0 else "N/A",
        'Casos CON intervención': n_treated,
        'Casos CON intervención (%)': f"{n_treated/n_cases:.2%}",
        'Casos SIN intervención': n_untreated,
        'Casos SIN intervención (%)': f"{n_untreated/n_cases:.2%}",
        'Eventos por caso (promedio)': round(n_events/n_cases, 2),
        'Eventos por caso (mediana)': round(events_per_case.median(), 2),
        'Duración promedio (días)': round(durations.mean(), 2),
        'Duración mediana (días)': round(durations.median(), 2),
        'Duración mínima (días)': round(durations.min(), 2),
        'Duración máxima (días)': round(durations.max(), 2),
        'Duración del log (días)': log_duration_days
    }
    
    # Crear diccionario en formato EDA (para summary_statistics.csv)
    metrics_eda_format = {
        'Métrica': [
            'Número de casos',
            'Número de eventos',
            'Actividades únicas',
            'Recursos únicos',
            'Casos terminados',
            'Casos en curso (sin terminar)',
            'Casos exitosos (de terminados)',
            'Casos no exitosos/fracaso (de terminados)',
            'Casos CON intervención (T=1)',
            'Casos SIN intervención (T=0)',
            'Eventos por caso (promedio)',
            'Eventos por caso (mediana)',
            'Duración promedio (días)',
            'Duración mediana (días)',
            'Duración mínima (días)',
            'Duración máxima (días)',
            'Duración del log (días)'
        ],
        'Valor': [
            f"{n_cases:,}",
            f"{n_events:,}",
            n_activities,
            n_resources,
            f"{completed:,} ({completed/n_cases:.2%})" if n_cases > 0 else "N/A",
            f"{ongoing:,} ({ongoing/n_cases:.2%})" if n_cases > 0 else "N/A",
            f"{n_success:,} ({n_success/completed:.2%} de terminados)" if completed > 0 else "N/A",
            f"{n_failed:,} ({n_failed/completed:.2%} de terminados)" if completed > 0 else "N/A",
            f"{n_treated:,} ({n_treated/n_cases:.2%})" if n_cases > 0 else "N/A",
            f"{n_untreated:,} ({n_untreated/n_cases:.2%})" if n_cases > 0 else "N/A",
            f"{n_events/n_cases:.2f}",
            f"{events_per_case.median():.2f}",
            f"{durations.mean():.2f}",
            f"{durations.median():.2f}",
            f"{durations.min():.2f}",
            f"{durations.max():.2f}",
            f"{log_duration_days}"
        ]
    }
    
    print(f"\n--- {name} ---")
    print(f"Número de casos: {n_cases:,}")
    print(f"Número de eventos: {n_events:,}")
    print(f"Actividades únicas: {n_activities}")
    print(f"Recursos únicos: {n_resources}")
    print(f"Casos terminados: {completed:,} ({completed/n_cases:.2%})")
    print(f"Casos en curso: {ongoing:,} ({ongoing/n_cases:.2%})")
    if completed > 0:
        print(f"Casos exitosos (de terminados): {n_success:,} ({n_success/completed:.2%})")
        print(f"Casos fracaso (de terminados): {n_failed:,} ({n_failed/completed:.2%})")
    print(f"Casos CON intervención: {n_treated:,} ({n_treated/n_cases:.2%})")
    print(f"Casos SIN intervención: {n_untreated:,} ({n_untreated/n_cases:.2%})")
    print(f"Eventos por caso (promedio): {n_events/n_cases:.2f}")
    print(f"Eventos por caso (mediana): {events_per_case.median():.2f}")
    print(f"Duración promedio (días): {durations.mean():.2f}")
    print(f"Duración mediana (días): {durations.median():.2f}")
    print(f"Duración mínima (días): {durations.min():.2f}")
    print(f"Duración máxima (días): {durations.max():.2f}")
    print(f"Duración del log (días): {log_duration_days}")
    
    return metrics, metrics_eda_format

if __name__ == "__main__":
    # Calcular rutas absolutas basadas en la ubicación del script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/benchmark/preprocess/
    src_dir = os.path.dirname(os.path.dirname(script_dir))  # src/
    project_root = os.path.dirname(src_dir)  # project root
    
    # Directorio donde están los archivos procesados
    base_dir = os.path.join(project_root, "logs", "BPI2017", "processed")
    
    train_path = os.path.join(base_dir, "bpi2017_train.csv")
    test_path = os.path.join(base_dir, "bpi2017_test.csv")
    
    print("="*80)
    print("📊 ESTADÍSTICAS DEL SPLIT TRAIN/TEST")
    print("="*80)
    print(f"📂 Project root: {project_root}")
    print(f"📂 Base dir: {base_dir}")
    print(f"📂 Train file: {train_path}")
    print(f"📂 Test file: {test_path}")
    print()
    
    # Verificar que los archivos existan
    if not os.path.exists(train_path):
        print(f"❌ Error: No se encontró el archivo: {train_path}")
        if os.path.exists(base_dir):
            print(f"   Archivos disponibles en {base_dir}:")
            for f in os.listdir(base_dir):
                print(f"     - {f}")
        exit(1)
    
    if not os.path.exists(test_path):
        print(f"❌ Error: No se encontró el archivo: {test_path}")
        exit(1)
    
    print("🔄 Cargando archivos...")
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    
    print(f"✅ Train: {len(train_df):,} eventos")
    print(f"✅ Test: {len(test_df):,} eventos")
    print()
    
    # Calcular métricas
    train_metrics, train_eda_format = calculate_metrics(train_df, "TRAIN SET (70%)")
    test_metrics, test_eda_format = calculate_metrics(test_df, "TEST SET (30%)")
    
    # Crear DataFrame y transponer (métricas como filas, datasets como columnas)
    stats_df = pd.DataFrame([train_metrics, test_metrics])
    
    # Transponer: las métricas serán las filas y los datasets las columnas
    stats_df_transposed = stats_df.set_index('Dataset').T
    stats_df_transposed.index.name = 'Métrica'
    stats_df_transposed.columns.name = None
    
    # Guardar estadísticas en CSV (transpuesto)
    stats_path = os.path.join(base_dir, "split_statistics.csv")
    stats_df_transposed.to_csv(stats_path)
    
    # Generar archivo summary_statistics.csv unificado (formato EDA combinado)
    # Crear DataFrame con métricas como filas y Train/Test como columnas
    summary_combined = pd.DataFrame({
        'Métrica': train_eda_format['Métrica'],
        'Train Set (70%)': train_eda_format['Valor'],
        'Test Set (30%)': test_eda_format['Valor']
    })
    
    summary_path = os.path.join(base_dir, "summary_statistics.csv")
    summary_combined.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("✅ ESTADÍSTICAS GUARDADAS")
    print("="*80)
    print(f"📄 Archivo transpuesto: {stats_path}")
    print(f"   - {len(stats_df_transposed)} métricas (filas)")
    print(f"   - {len(stats_df_transposed.columns)} datasets (columnas: Train y Test)")
    print(f"\n📄 Summary unificado (formato EDA): {summary_path}")
    print(f"   - {len(summary_combined)} métricas (filas)")
    print(f"   - 2 columnas: Train Set (70%) y Test Set (30%)")