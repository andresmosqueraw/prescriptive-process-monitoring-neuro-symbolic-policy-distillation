#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para calcular el estado parcial del proceso usando ongoing-bps-state-short-term.
Genera el archivo JSON con el estado parcial del proceso en un punto de corte temporal.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Verificar si estamos en un entorno virtual
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        print("⚠️  No se detectó entorno virtual activo.")
        print(f"💡 Ejecuta: source {script_dir}/venv/bin/activate")
        print("   O ejecuta el script con: venv/bin/python compute_state.py")
        print()

# Agregar path de ongoing-bps-state
ongoing_bps_path = "/home/andrew/Documents/asistencia-graduada-phd-oscar/paper1/repos-asis-online-predictivo/whats-coming-next-short-term-simulation-of-business-processes-from-current-state/ongoing-bps-state-short-term"

if not os.path.exists(ongoing_bps_path):
    print(f"❌ No se encontró la carpeta ongoing-bps-state-short-term en: {ongoing_bps_path}")
    sys.exit(1)

if ongoing_bps_path not in sys.path:
    sys.path.insert(0, ongoing_bps_path)

# Importar después de agregar paths
from src.runner import run_process_state_and_simulation

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

def get_log_name_from_path(log_path):
    """Extrae el nombre del log desde la ruta"""
    return os.path.splitext(os.path.basename(log_path))[0]

def compute_cut_points(
    log_df: pd.DataFrame,
    horizon_days: int,
    *,
    strategy: str = "fixed",
    fixed_cut: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[pd.Timestamp]:
    """
    Return a list of cut-off timestamps according to *strategy*.
    
    Strategies:
    ----------
    fixed
        Exactly one timestamp, taken from *fixed_cut*.
        If fixed_cut is None, uses the last event timestamp.
    wip3
        Three timestamps where the Work-in-Process equals 10 %, 50 %, and
        90 % of the maximum observed WiP.
    segment10
        Ten timestamps: drop the first and last *horizon* and divide the
        remaining interval into ten equal segments; pick one random moment
        from each segment.
    """
    if strategy == "fixed":
        if fixed_cut is None:
            # Usar último evento del log
            return [log_df["end_time"].max()]
        return [pd.to_datetime(fixed_cut, utc=True)]
    
    rng = rng or np.random.default_rng()
    
    first_ts = log_df["StartTime"].min()
    last_ts  = log_df["EndTime"].max()
    
    safe_start = first_ts + pd.Timedelta(days=horizon_days)
    safe_end   = last_ts  - pd.Timedelta(days=horizon_days)
    if safe_start >= safe_end:
        raise ValueError("the event log is shorter than twice the horizon")
    
    # helper: active cases at a given time
    # Asegurar que las columnas estén presentes
    if "CaseId" not in log_df.columns or "StartTime" not in log_df.columns or "EndTime" not in log_df.columns:
        raise ValueError("El log debe tener columnas CaseId, StartTime, EndTime después del mapeo")
    
    case_bounds = log_df.groupby("CaseId").agg(
        start=("StartTime", "min"),
        end=("EndTime",   "max"),
    )
    def active_cases_at(ts: pd.Timestamp) -> int:
        mask = (case_bounds["start"] <= ts) & (case_bounds["end"] > ts)
        return int(mask.sum())
    
    if strategy == "wip3":
        # evaluate WiP only at case arrival moments
        arrivals = case_bounds["start"].sort_values()
        wip_series = pd.Series(
            {ts: active_cases_at(ts) for ts in arrivals}
        )
        max_wip = wip_series.max()
        targets = [int(round(max_wip * q)) for q in (0.10, 0.50, 0.90)]
        
        cuts: list[pd.Timestamp] = []
        for tgt in targets:
            exact = wip_series[wip_series == tgt]
            if not exact.empty:
                cuts.append(exact.index[0])
                continue
            greater = wip_series[wip_series > tgt]
            if not greater.empty:
                cuts.append(greater.index[0])
                continue
            cuts.append(wip_series.index[0]) 
        return cuts
    
    if strategy == "segment10":
        span = safe_end - safe_start
        segment_length = span / 10
        cuts: list[pd.Timestamp] = []
        for i in range(10):
            seg_start = safe_start + i * segment_length
            jitter = rng.uniform(0, segment_length.total_seconds())
            cuts.append(seg_start + pd.Timedelta(seconds=float(jitter)))
        return cuts
    
    raise ValueError(f"unknown cut strategy: {strategy}")

def compute_state(config=None):
    """
    Calcula el estado parcial del proceso.
    
    Args:
        config: Diccionario de configuración (si es None, se carga desde config.yaml)
    
    Returns:
        str: Ruta al archivo de estado generado, o None si falló
    """
    print("=" * 80)
    print("🔧 CÁLCULO DE ESTADO PARCIAL DEL PROCESO")
    print("=" * 80)
    
    # Cargar configuración
    if config is None:
        config = load_config()
        if config is None:
            return None
    
    # Obtener configuración
    ongoing_config = config.get("ongoing_config", {})
    log_config = config.get("log_config", {})
    script_config = config.get("script_config", {})
    
    # Obtener rutas de archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    # Directorio de salida para estado parcial
    output_dir = script_config.get("state_output_dir")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "generado-state")
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener ruta del log
    log_path = log_config.get("log_path")
    if not log_path:
        print("❌ Error: No se especificó log_path en config.yaml")
        return None
    
    # Si es una ruta relativa, hacerla relativa al directorio base (nuevo/)
    if not os.path.isabs(log_path):
        log_path = os.path.join(base_dir, log_path)
    
    # Obtener nombre del log
    log_name = get_log_name_from_path(log_path)
    
    # Rutas de BPMN y JSON
    simod_output_dir = os.path.join(base_dir, "data", "generado-simod")
    bpmn_path_simod = os.path.join(simod_output_dir, f"{log_name}.bpmn")
    json_path_simod = os.path.join(simod_output_dir, f"{log_name}.json")
    
    if os.path.exists(bpmn_path_simod) and os.path.exists(json_path_simod):
        bpmn_path = bpmn_path_simod
        json_path = json_path_simod
    else:
        bpmn_path = os.path.join(base_dir, f"{log_name}.bpmn")
        json_path = os.path.join(base_dir, f"{log_name}.json")
    
    # Verificar que todos los archivos existan
    files_to_check = {
        "Event Log": log_path,
        "BPMN Model": bpmn_path,
        "JSON Parameters": json_path
    }
    
    print("\n📋 Verificando archivos necesarios...")
    for file_type, path in files_to_check.items():
        if not os.path.exists(path):
            print(f"❌ Archivo no encontrado ({file_type}): {path}")
            return None
        else:
            print(f"✅ {file_type}: {path}")
    
    # Obtener parámetros de configuración
    column_mapping = ongoing_config.get("column_mapping")
    
    # Si column_mapping es null en ongoing_config, usar el de log_config
    if column_mapping is None:
        column_mapping = log_config.get("column_mapping")
    
    # Convertir column_mapping a formato para pandas (csv_name -> standard_name)
    if column_mapping and isinstance(column_mapping, dict):
        csv_to_standard = {
            column_mapping.get("case", "caseid"): "CaseId",
            column_mapping.get("activity", "task"): "Activity",
            column_mapping.get("resource", "user"): "Resource",
            column_mapping.get("start_time", "start_timestamp"): "StartTime",
            column_mapping.get("end_time", "end_timestamp"): "EndTime"
        }
        column_mapping_json = json.dumps(csv_to_standard)
    elif column_mapping is None:
        csv_to_standard = {}
        column_mapping_json = None
    else:
        csv_to_standard = {}
        column_mapping_json = column_mapping
    
    # Leer el log para calcular puntos de corte
    print("\n📊 Leyendo log de eventos para calcular puntos de corte...")
    log_df = pd.read_csv(log_path)
    
    # Aplicar mapeo de columnas
    if csv_to_standard:
        log_df = log_df.rename(columns=csv_to_standard)
    
    # Convertir timestamps
    log_df['StartTime'] = pd.to_datetime(log_df['StartTime'], utc=True)
    log_df['EndTime'] = pd.to_datetime(log_df['EndTime'], utc=True)
    
    # Obtener estrategia de puntos de corte
    cut_strategy = ongoing_config.get("cut_strategy", "fixed")
    fixed_cut = ongoing_config.get("start_time")  # Para estrategia "fixed"
    horizon_days = ongoing_config.get("horizon_days", 7)
    
    print(f"\n📅 Estrategia de puntos de corte: {cut_strategy}")
    
    # Calcular puntos de corte
    try:
        cut_points = compute_cut_points(
            log_df=log_df,
            horizon_days=horizon_days,
            strategy=cut_strategy,
            fixed_cut=fixed_cut,
            rng=None
        )
        print(f"✅ Calculados {len(cut_points)} puntos de corte")
        for i, cut in enumerate(cut_points, 1):
            print(f"   {i}. {cut.isoformat()}")
    except Exception as e:
        print(f"❌ Error calculando puntos de corte: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Cambiar al directorio de salida
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    generated_files = []
    
    try:
        # Procesar cada punto de corte
        for i, cut_point in enumerate(cut_points, 1):
            print(f"\n{'='*80}")
            print(f"📅 Procesando punto de corte {i}/{len(cut_points)}: {cut_point.isoformat()}")
            print(f"{'='*80}")
            
            # Convertir pd.Timestamp a string ISO para run_process_state_and_simulation
            start_time_str = cut_point.isoformat()
            
            # Calcular estado (sin simulación)
            result = run_process_state_and_simulation(
                event_log=log_path,
                bpmn_model=bpmn_path,
                bpmn_parameters=json_path,
                start_time=start_time_str,
                column_mapping=column_mapping_json,
                simulate=False,  # Solo calcular estado
                total_cases=ongoing_config.get("total_cases", 20)
            )
            
            # Leer output.json que se creó en el directorio de salida
            output_json_path = os.path.join(output_dir, "output.json")
            
            if os.path.exists(output_json_path):
                # Generar nombre de archivo con sufijo del punto de corte
                suffix = cut_point.strftime("%Y%m%d_%H%M%S")
                state_file = os.path.join(output_dir, f"{log_name}_process_state_{suffix}.json")
                
                # Copiar y renombrar
                import shutil
                shutil.copy2(output_json_path, state_file)
                os.remove(output_json_path)  # Eliminar output.json temporal
                
                print(f"✅ Estado calculado y guardado en: {state_file}")
                generated_files.append(state_file)
            else:
                print(f"⚠️  No se encontró output.json en: {output_json_path}")
                if result is None:
                    print(f"❌ Error: result es None y no se encontró output.json")
        
        print(f"\n{'='*80}")
        print(f"✅ Proceso completado: {len(generated_files)}/{len(cut_points)} estados generados")
        print(f"{'='*80}")
        print(f"\n📁 Estados parciales guardados en: {output_dir}")
        for f in generated_files:
            print(f"   • {os.path.basename(f)}")
        
        return generated_files if len(generated_files) > 0 else None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restaurar directorio original
        os.chdir(original_cwd)

def main():
    """Función principal para ejecutar desde línea de comandos"""
    result = compute_state()
    if result:
        if isinstance(result, list):
            print(f"\n🎉 ¡{len(result)} estados parciales calculados exitosamente!")
        else:
            print("\n🎉 ¡Estado parcial calculado exitosamente!")
        sys.exit(0)
    else:
        print("\n❌ El cálculo del estado falló")
        sys.exit(1)

if __name__ == "__main__":
    main()

