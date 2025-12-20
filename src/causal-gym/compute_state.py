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
import gzip
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logging import setup_logger

# Configurar logger
logger = setup_logger(__name__)

# Verificar si estamos en un entorno virtual
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        logger.warning("No se detectó entorno virtual activo.")
        logger.info(f"Ejecuta: source {script_dir}/venv/bin/activate")
        logger.info("O ejecuta el script con: venv/bin/python compute_state.py")

# Cargar configuración para obtener rutas
temp_config = load_config()
if temp_config is None:
    logger.error("No se pudo cargar la configuración desde configs/config.yaml")
    sys.exit(1)

# Obtener ruta a ongoing-bps-state-short-term desde config.yaml
if not temp_config.get("external_repos"):
    logger.error("No se encontró la sección 'external_repos' en configs/config.yaml")
    logger.error("Agrega la siguiente sección a tu config.yaml:")
    logger.error("  external_repos:")
    logger.error("    ongoing_bps_state_path: /ruta/a/ongoing-bps-state-short-term")
    sys.exit(1)

ongoing_bps_path = temp_config["external_repos"].get("ongoing_bps_state_path")

if not ongoing_bps_path:
    logger.error("No se encontró 'ongoing_bps_state_path' en configs/config.yaml")
    logger.error("Configura la ruta en configs/config.yaml:")
    logger.error("  external_repos:")
    logger.error("    ongoing_bps_state_path: /ruta/a/ongoing-bps-state-short-term")
    sys.exit(1)

if not os.path.exists(ongoing_bps_path):
    logger.error(f"La ruta configurada no existe: {ongoing_bps_path}")
    logger.error("Verifica que la ruta en configs/config.yaml sea correcta")
    sys.exit(1)

if ongoing_bps_path not in sys.path:
    sys.path.insert(0, ongoing_bps_path)

# Importar después de agregar paths
from src.runner import run_process_state_and_simulation

# get_log_name_from_path ahora se importa de utils.config

def compute_cut_points(
    log_df: pd.DataFrame,
    horizon_days: int,
    *,
    strategy: str = "fixed",
    fixed_cut: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[pd.Timestamp]:
    """
    Return a list of cut-off timestamps according to *strategy*.
    
    Args:
        log_df: DataFrame con el log de eventos
        horizon_days: Días de horizonte para calcular puntos seguros
        strategy: Estrategia para calcular puntos de corte ("fixed", "wip3", "segment10")
        fixed_cut: Timestamp fijo para estrategia "fixed"
        rng: Generador de números aleatorios (opcional)
    
    Returns:
        Lista de timestamps de puntos de corte
    
    Raises:
        ValueError: Si el log es muy corto o faltan columnas necesarias
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

def compute_state(config: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """
    Calcula el estado parcial del proceso.
    
    Args:
        config: Diccionario de configuración (si es None, se carga desde config.yaml)
    
    Returns:
        Lista de rutas a archivos de estado generados, o None si falló
    """
    logger.info("=" * 80)
    logger.info("CÁLCULO DE ESTADO PARCIAL DEL PROCESO")
    logger.info("=" * 80)
    
    # Cargar configuración
    if config is None:
        config = load_config()
        if config is None:
            logger.error("No se pudo cargar la configuración")
            return None
    
    # Obtener configuración
    ongoing_config = config.get("ongoing_config", {})
    log_config = config.get("log_config", {})
    script_config = config.get("script_config", {})
    
    # Obtener ruta del log primero para extraer el nombre
    # Encontrar el directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = src/causal-gym/
    src_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    log_path = log_config.get("log_path")
    if not log_path:
        logger.error("No se especificó log_path en config.yaml")
        return None
    
    # Si es una ruta relativa, hacerla relativa al directorio raíz del proyecto
    if not os.path.isabs(log_path):
        log_path = os.path.join(project_root, log_path)
    
    # Obtener nombre del log
    log_name = get_log_name_from_path(log_path)
    
    # Directorio de salida para estado parcial (incluyendo nombre del log)
    state_output_dir_base = script_config.get("state_output_dir")
    output_dir = build_output_path(state_output_dir_base, log_name, "state", default_base="data")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Rutas de BPMN y JSON - usar script_config.output_dir (donde Simod guarda los archivos)
    # Incluir nombre del log en la ruta
    simod_output_dir_base = script_config.get("output_dir")
    simod_output_dir = build_output_path(simod_output_dir_base, log_name, "simod", default_base="data")
    
    bpmn_path = os.path.join(simod_output_dir, f"{log_name}.bpmn")
    json_path = os.path.join(simod_output_dir, f"{log_name}.json")
    
    # Verificar que todos los archivos existan
    files_to_check = {
        "Event Log": log_path,
        "BPMN Model": bpmn_path,
        "JSON Parameters": json_path
    }
    
    logger.info("Verificando archivos necesarios...")
    for file_type, path in files_to_check.items():
        if not os.path.exists(path):
            logger.error(f"Archivo no encontrado ({file_type}): {path}")
            return None
        else:
            logger.info(f"{file_type}: {path}")
    
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
    logger.info("Leyendo log de eventos para calcular puntos de corte...")
    
    # Detectar formato del log
    log_ext = os.path.splitext(log_path)[1].lower()
    if log_ext == '.gz':
        log_ext = os.path.splitext(os.path.splitext(log_path)[0])[1].lower()
    
    if log_ext == '.xes':
        # Leer archivo XES usando pm4py
        try:
            import pm4py
            if log_path.endswith('.gz'):
                # Descomprimir temporalmente
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp_file:
                    with gzip.open(log_path, 'rb') as f_in:
                        tmp_file.write(f_in.read())
                    tmp_path = tmp_file.name
                log_df = pm4py.convert_to_dataframe(pm4py.read_xes(tmp_path))
                os.unlink(tmp_path)
            else:
                log_df = pm4py.convert_to_dataframe(pm4py.read_xes(log_path))
            
            # Mapear columnas estándar de XES a formato esperado
            xes_to_standard = {
                'case:concept:name': 'CaseId',
                'concept:name': 'Activity',
                'org:resource': 'Resource',
                'time:timestamp': 'EndTime',  # XES generalmente tiene un timestamp
            }
            
            # Renombrar columnas si existen
            for xes_col, std_col in xes_to_standard.items():
                if xes_col in log_df.columns:
                    log_df = log_df.rename(columns={xes_col: std_col})
            
            # Si no hay StartTime, usar EndTime como StartTime
            if 'StartTime' not in log_df.columns and 'EndTime' in log_df.columns:
                log_df['StartTime'] = log_df['EndTime']
            
            # Para XES, no aplicar mapeo de CSV (ya está mapeado)
            csv_to_standard = {}
        except ImportError:
            logger.error("pm4py no está instalado. Instala con: pip install pm4py")
            logger.error("O convierte el log XES a CSV primero")
            return None
        except Exception as e:
            logger.error(f"Error leyendo archivo XES: {e}", exc_info=True)
            return None
    else:
        # Leer archivo CSV
        log_df = pd.read_csv(log_path)
        
        # Aplicar mapeo de columnas solo para CSV
        if csv_to_standard:
            log_df = log_df.rename(columns=csv_to_standard)
    
    # Convertir timestamps
    log_df['StartTime'] = pd.to_datetime(log_df['StartTime'], utc=True)
    log_df['EndTime'] = pd.to_datetime(log_df['EndTime'], utc=True)
    
    # Obtener estrategia de puntos de corte
    cut_strategy = ongoing_config.get("cut_strategy", "fixed")
    fixed_cut = ongoing_config.get("start_time")  # Para estrategia "fixed"
    horizon_days = ongoing_config.get("horizon_days", 7)
    
    logger.info(f"Estrategia de puntos de corte: {cut_strategy}")
    
    # Calcular puntos de corte
    try:
        cut_points = compute_cut_points(
            log_df=log_df,
            horizon_days=horizon_days,
            strategy=cut_strategy,
            fixed_cut=fixed_cut,
            rng=None
        )
        logger.info(f"Calculados {len(cut_points)} puntos de corte")
        for i, cut in enumerate(cut_points, 1):
            logger.info(f"  {i}. {cut.isoformat()}")
    except Exception as e:
        logger.error(f"Error calculando puntos de corte: {e}", exc_info=True)
        return None
    
    # Cambiar al directorio de salida
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    generated_files = []
    
    try:
        # Procesar cada punto de corte
        for i, cut_point in enumerate(cut_points, 1):
            logger.info(f"{'='*80}")
            logger.info(f"Procesando punto de corte {i}/{len(cut_points)}: {cut_point.isoformat()}")
            logger.info(f"{'='*80}")
            
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
                
                logger.info(f"Estado calculado y guardado en: {state_file}")
                generated_files.append(state_file)
            else:
                logger.warning(f"No se encontró output.json en: {output_json_path}")
                if result is None:
                    logger.error("result es None y no se encontró output.json")
        
        logger.info(f"{'='*80}")
        logger.info(f"Proceso completado: {len(generated_files)}/{len(cut_points)} estados generados")
        logger.info(f"{'='*80}")
        logger.info(f"Estados parciales guardados en: {output_dir}")
        for f in generated_files:
            logger.info(f"  • {os.path.basename(f)}")
        
        return generated_files if len(generated_files) > 0 else None
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None
    finally:
        # Restaurar directorio original
        os.chdir(original_cwd)

def main() -> None:
    """Función principal para ejecutar desde línea de comandos"""
    result = compute_state()
    if result:
        if isinstance(result, list):
            logger.info(f"¡{len(result)} estados parciales calculados exitosamente!")
        else:
            logger.info("¡Estado parcial calculado exitosamente!")
        sys.exit(0)
    else:
        logger.error("El cálculo del estado falló")
        sys.exit(1)

if __name__ == "__main__":
    main()

