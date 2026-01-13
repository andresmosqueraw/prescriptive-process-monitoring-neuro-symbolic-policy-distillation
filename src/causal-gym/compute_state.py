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
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logger_utils import setup_logger

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

def compute_state(config: Optional[Dict[str, Any]] = None, log_path_override: Optional[str] = None, force: bool = False) -> Optional[List[str]]:
    """
    Calcula el estado parcial del proceso.
    
    Args:
        config: Diccionario de configuración (si es None, se carga desde config.yaml)
        log_path_override: Ruta al log a usar (sobrescribe log_config.log_path)
    
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
    
    # Usar log_path_override si se proporciona, sino usar log_config
    if log_path_override:
        log_path = log_path_override
        if not os.path.isabs(log_path):
            log_path = os.path.join(project_root, log_path)
    else:
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
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    # Verificar si ya existen estados parciales calculados (a menos que force=True)
    if not force and os.path.exists(output_dir):
        existing_state_files = [
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith(f"{log_name}_process_state_") and f.endswith(".json")
        ]
        if existing_state_files:
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ ESTADOS PARCIALES YA EXISTEN - OMITIENDO CÁLCULO")
            logger.info("=" * 80)
            logger.info(f"📁 Directorio: {output_dir}")
            logger.info(f"   Encontrados {len(existing_state_files)} archivo(s) de estado:")
            for f in sorted(existing_state_files):
                file_size = os.path.getsize(f) / 1024  # KB
                logger.info(f"     ✅ {os.path.basename(f)} ({file_size:.2f} KB)")
            logger.info("")
            logger.info("💡 Para recalcular, elimina los archivos primero o usa --force")
            logger.info("")
            return existing_state_files
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Rutas de BPMN y JSON - usar script_config.output_dir (donde Simod guarda los archivos)
    # Incluir nombre del log en la ruta
    simod_output_dir_base = script_config.get("output_dir")
    simod_output_dir = build_output_path(simod_output_dir_base, log_name, "simod", default_base="data")
    if not os.path.isabs(simod_output_dir):
        simod_output_dir = os.path.join(project_root, simod_output_dir)
    
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
    
    # Construir mapeo de columnas para JSON (para ongoing-bps-state-short-term)
    # La librería espera un mapeo: {nombre_columna_csv: nombre_estándar}
    # Por ejemplo: {"case:concept:name": "CaseId", "concept:name": "Activity", ...}
    if column_mapping and isinstance(column_mapping, dict):
        # Convertir el mapeo del config.yaml al formato que espera la librería
        # config.yaml tiene: {case: "case:concept:name", activity: "concept:name", ...}
        # La librería espera: {"case:concept:name": "CaseId", "concept:name": "Activity", ...}
        standard_mapping = {
            'case': 'CaseId',
            'activity': 'Activity',
            'resource': 'Resource',
            'start_time': 'StartTime',
            'end_time': 'EndTime'
        }
        
        csv_to_standard_for_lib = {}
        for config_key, standard_name in standard_mapping.items():
            if config_key in column_mapping:
                csv_col_name = column_mapping[config_key]
                # Si start_time y end_time apuntan a la misma columna, mapear solo a EndTime
                # La librería creará StartTime desde EndTime si es necesario
                if config_key == 'start_time' and csv_col_name == column_mapping.get('end_time'):
                    # Ambos apuntan a la misma columna, mapear solo a EndTime
                    csv_to_standard_for_lib[csv_col_name] = 'EndTime'
                elif config_key == 'end_time' and csv_col_name == column_mapping.get('start_time'):
                    # Ya se mapeó arriba como EndTime, saltar
                    continue
                else:
                    csv_to_standard_for_lib[csv_col_name] = standard_name
        
        # Si start_time y end_time apuntan a la misma columna, necesitamos mapear también a StartTime
        # Pero la librería solo puede mapear una columna a un nombre estándar, así que mapeamos a EndTime
        # y la librería debería manejar StartTime internamente
        # Sin embargo, si la librería requiere StartTime explícitamente, necesitamos duplicar la columna
        # Por ahora, mapeamos solo a EndTime y confiamos en que la librería maneje StartTime
        
        column_mapping_json = json.dumps(csv_to_standard_for_lib)
        logger.info(f"Mapeo de columnas para librería: {column_mapping_json}")
    else:
        column_mapping_json = None
    
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
        
        # Construir mapeo de columnas desde config.yaml a nombres estándar
        # Mapeo esperado: case -> CaseId, activity -> Activity, resource -> Resource,
        #                 start_time -> StartTime, end_time -> EndTime
        csv_to_standard = {}
        start_time_col = None
        end_time_col = None
        
        if column_mapping:
            # Mapear nombres del config a nombres estándar
            standard_mapping = {
                'case': 'CaseId',
                'activity': 'Activity',
                'resource': 'Resource',
                'start_time': 'StartTime',
                'end_time': 'EndTime'
            }
            
            for config_key, standard_name in standard_mapping.items():
                if config_key in column_mapping:
                    csv_col_name = column_mapping[config_key]
                    if csv_col_name in log_df.columns:
                        # Guardar referencias a las columnas de tiempo
                        if config_key == 'start_time':
                            start_time_col = csv_col_name
                        elif config_key == 'end_time':
                            end_time_col = csv_col_name
                        
                        # Si start_time y end_time apuntan a la misma columna,
                        # solo mapear a EndTime y luego crear StartTime desde EndTime
                        if config_key == 'start_time' and csv_col_name == column_mapping.get('end_time'):
                            # Ambos apuntan a la misma columna, mapear solo a EndTime
                            csv_to_standard[csv_col_name] = 'EndTime'
                            logger.debug(f"Mapeando columna CSV '{csv_col_name}' -> 'EndTime' (start_time y end_time son iguales)")
                        elif config_key == 'end_time' and csv_col_name == column_mapping.get('start_time'):
                            # Ya se mapeó arriba, saltar
                            continue
                        else:
                            csv_to_standard[csv_col_name] = standard_name
                            logger.debug(f"Mapeando columna CSV '{csv_col_name}' -> '{standard_name}'")
        
        # Aplicar mapeo de columnas
        if csv_to_standard:
            log_df = log_df.rename(columns=csv_to_standard)
            logger.info(f"Columnas mapeadas: {list(csv_to_standard.keys())} -> {list(csv_to_standard.values())}")
        
        # Si StartTime y EndTime apuntan a la misma columna, crear StartTime desde EndTime
        if 'StartTime' not in log_df.columns and 'EndTime' in log_df.columns:
            log_df['StartTime'] = log_df['EndTime'].copy()
            logger.info("StartTime creado desde EndTime (mismo timestamp)")
        
        # Verificar que StartTime y EndTime existan después del mapeo
        if 'StartTime' not in log_df.columns:
            logger.error(f"Columna 'StartTime' no encontrada después del mapeo. Columnas disponibles: {list(log_df.columns)}")
            raise ValueError("Columna 'StartTime' no encontrada. Verifica el mapeo de columnas en config.yaml")
        if 'EndTime' not in log_df.columns:
            logger.error(f"Columna 'EndTime' no encontrada después del mapeo. Columnas disponibles: {list(log_df.columns)}")
            raise ValueError("Columna 'EndTime' no encontrada. Verifica el mapeo de columnas en config.yaml")
    
    # Convertir timestamps (usar format='mixed' para manejar diferentes formatos)
    log_df['StartTime'] = pd.to_datetime(log_df['StartTime'], utc=True, format='mixed', errors='coerce')
    log_df['EndTime'] = pd.to_datetime(log_df['EndTime'], utc=True, format='mixed', errors='coerce')
    
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
    
    # Crear un CSV temporal con columnas renombradas si es necesario
    # Esto es necesario porque la librería requiere StartTime y EndTime explícitos
    temp_log_path = None
    try:
        # Si start_time y end_time apuntan a la misma columna, crear CSV temporal
        if column_mapping and column_mapping.get('start_time') == column_mapping.get('end_time'):
            logger.info("Creando CSV temporal con columnas renombradas (start_time y end_time son iguales)...")
            import tempfile
            import shutil
            
            # Leer el CSV original
            temp_log_df = pd.read_csv(log_path)
            
            # Aplicar el mapeo de columnas
            if csv_to_standard:
                temp_log_df = temp_log_df.rename(columns=csv_to_standard)
            
            # Crear StartTime desde EndTime si no existe
            if 'StartTime' not in temp_log_df.columns and 'EndTime' in temp_log_df.columns:
                temp_log_df['StartTime'] = temp_log_df['EndTime'].copy()
            
            # Guardar CSV temporal
            temp_log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_log_path = temp_log_file.name
            temp_log_df.to_csv(temp_log_path, index=False)
            temp_log_file.close()
            
            logger.info(f"CSV temporal creado: {temp_log_path}")
            logger.info(f"Columnas en CSV temporal: {list(temp_log_df.columns)}")
            
            # No pasar column_mapping a la librería ya que las columnas ya están renombradas
            column_mapping_for_lib = None
            event_log_to_use = temp_log_path
        else:
            event_log_to_use = log_path
            column_mapping_for_lib = column_mapping_json
        
        # Procesar cada punto de corte
        for i, cut_point in enumerate(cut_points, 1):
            logger.info(f"{'='*80}")
            logger.info(f"Procesando punto de corte {i}/{len(cut_points)}: {cut_point.isoformat()}")
            logger.info(f"{'='*80}")
            
            # Convertir pd.Timestamp a string ISO para run_process_state_and_simulation
            start_time_str = cut_point.isoformat()
            
            # Calcular estado (sin simulación)
            result = run_process_state_and_simulation(
                event_log=event_log_to_use,
                bpmn_model=bpmn_path,
                bpmn_parameters=json_path,
                start_time=start_time_str,
                column_mapping=column_mapping_for_lib,
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
        # Limpiar archivo temporal si se creó
        if temp_log_path and os.path.exists(temp_log_path):
            try:
                os.unlink(temp_log_path)
                logger.debug(f"Archivo temporal eliminado: {temp_log_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {temp_log_path}: {e}")
        
        # Restaurar directorio original
        os.chdir(original_cwd)

def main() -> None:
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(description='Calcular estado parcial del proceso')
    parser.add_argument('--train', action='store_true',
                       help='Usar archivo de train procesado (bpi2017_train.csv)')
    parser.add_argument('--test', action='store_true',
                       help='Usar archivo de test procesado (bpi2017_test.csv)')
    parser.add_argument('--force', action='store_true',
                       help='Forzar recálculo incluso si ya existen estados parciales')
    args = parser.parse_args()
    
    # Encontrar directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    log_path_override = None
    if args.train:
        log_path_override = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_train.csv")
        logger.info(f"🎯 Modo TRAIN: Usando archivo de train: {log_path_override}")
    elif args.test:
        log_path_override = os.path.join(project_root, "logs", "BPI2017", "processed", "bpi2017_test.csv")
        logger.info(f"🎯 Modo TEST: Usando archivo de test: {log_path_override}")
    
    result = compute_state(log_path_override=log_path_override, force=args.force)
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

