#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para extraer BPMN y JSON desde un log usando Simod.
Recibe el log original y genera BPMN y JSON en la carpeta de salida configurada.
"""

import os
import sys
import subprocess
import gzip
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from utils.config import load_config
from utils.logging import setup_logger

# Configurar logger
logger = setup_logger(__name__)

def get_default_config() -> Dict[str, Any]:
    """
    Retorna configuración por defecto si no se encuentra el archivo.
    
    Returns:
        Diccionario con configuración por defecto
    """
    return {
        "log_config": {
            "column_mapping": {
                "case": "caseid",
                "activity": "task",
                "resource": "user",
                "start_time": "start_timestamp",
                "end_time": "end_timestamp"
            }
        },
        "simod_config": {
            "version": 5,
            "common": {"discover_data_attributes": True},
            "preprocessing": {"enable_time_concurrency_threshold": 0.0},
            "control_flow": {
                "optimization_metric": "two_gram_distance",
                "num_iterations": 10,
                "num_evaluations_per_iteration": 3,
                "gateway_probabilities": "discovery",
                "mining_algorithm": "sm2",
                "epsilon": [0.05, 0.5],
                "eta": [0.2, 0.7],
                "replace_or_joins": [True, False],
                "prioritize_parallelism": [True, False]
            },
            "resource_model": {
                "optimization_metric": "circadian_emd",
                "num_iterations": 5,
                "num_evaluations_per_iteration": 3,
                "discover_prioritization_rules": False,
                "discover_batching_rules": False,
                "resource_profiles": {
                    "discovery_type": "differentiated",
                    "granularity": 60,
                    "confidence": [0.6, 0.7],
                    "support": [0.05, 0.5],
                    "participation": 0.4
                }
            },
            "extraneous_activity_delays": {
                "discovery_method": "eclipse-aware",
                "num_iterations": 1
            }
        },
        "script_config": {
            "output_dir": None,
            "temp_dir_prefix": ".simod_temp",
            "docker": {
                "image": "nokal/simod",
                "user_id": None,
                "group_id": None
            }
        }
    }

def compress_log_to_gz(csv_path, gz_path):
    """Comprime un archivo CSV a formato .gz"""
    print(f"📦 Comprimiendo {csv_path} a {gz_path}...")
    with open(csv_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"✅ Log comprimido: {gz_path}")

def create_config_yaml(
    log_name: str,
    output_dir: str,
    simod_config: Dict[str, Any],
    log_config: Dict[str, Any],
    log_format: str = "csv"
) -> str:
    """
    Crea el archivo de configuración YAML para Simod.
    
    Args:
        log_name: Nombre del log (sin extensión)
        output_dir: Directorio donde guardar el archivo de configuración
        simod_config: Configuración de Simod desde config.yaml
        log_config: Configuración del log desde config.yaml
        log_format: Formato del log ("csv" o "xes")
    
    Returns:
        Ruta al archivo de configuración creado
    """
    config_path = os.path.join(output_dir, "configuration.yaml")
    
    # Determinar nombre del archivo según formato
    if log_format == "xes":
        train_log_path = f"./{log_name}.xes"
    else:
        train_log_path = f"./{log_name}.csv.gz"
    
    # Construir configuración para Simod usando los valores del config.yaml
    simod_yaml_config = {
        "version": simod_config.get("version", 5),
        "common": {
            "train_log_path": train_log_path,
            "log_ids": {
                "case": log_config["column_mapping"]["case"],
                "activity": log_config["column_mapping"]["activity"],
                "resource": log_config["column_mapping"]["resource"],
                "start_time": log_config["column_mapping"]["start_time"],
                "end_time": log_config["column_mapping"]["end_time"]
            },
            "discover_data_attributes": simod_config.get("common", {}).get("discover_data_attributes", True)
        },
        "preprocessing": simod_config.get("preprocessing", {}),
        "control_flow": simod_config.get("control_flow", {}),
        "resource_model": simod_config.get("resource_model", {}),
        "extraneous_activity_delays": simod_config.get("extraneous_activity_delays", {})
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(simod_yaml_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Archivo de configuración creado: {config_path}")
    return config_path

def run_simod_docker(
    input_path: str,
    output_path: str,
    config_file_name: str,
    docker_config: Dict[str, Any]
) -> bool:
    """
    Ejecuta Simod usando Docker.
    
    Args:
        input_path: Directorio de entrada montado en Docker
        output_path: Directorio de salida montado en Docker
        config_file_name: Nombre del archivo de configuración dentro del contenedor
        docker_config: Configuración de Docker desde config.yaml
    
    Returns:
        True si Simod se ejecutó exitosamente, False en caso contrario
    """
    config_inside_container = f"/usr/src/Simod/resources/{config_file_name}"
    
    # Obtener user_id y group_id de la configuración o usar valores por defecto
    user_id = docker_config.get("user_id") or os.getuid()
    group_id = docker_config.get("group_id") or os.getgid()
    docker_image = docker_config.get("image", "nokal/simod")
    
    docker_command = [
        "docker", "run", "--rm",
        "--user", f"{user_id}:{group_id}",
        "-v", f"{input_path}:/usr/src/Simod/resources",
        "-v", f"{output_path}:/usr/src/Simod/outputs",
        docker_image,
        "poetry", "run", "simod",
        "--configuration", config_inside_container
    ]
    
    logger.info("Ejecutando Simod con Docker")
    logger.debug(f"Comando Docker: {' '.join(docker_command)}")
    
    result = subprocess.run(docker_command, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Simod ejecutado exitosamente")
        if result.stdout:
            logger.debug(f"Salida de Simod: {result.stdout}")
        return True
    else:
        logger.error("Simod falló")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        return False

def find_and_copy_results(
    simod_output_dir: str,
    target_dir: str,
    log_name: str
) -> bool:
    """
    Busca los archivos BPMN y JSON generados por Simod y los copia a la carpeta objetivo.
    
    Args:
        simod_output_dir: Directorio donde Simod generó los resultados
        target_dir: Directorio destino donde copiar los archivos
        log_name: Nombre del log (sin extensión)
    
    Returns:
        True si se encontraron y copiaron los archivos, False en caso contrario
    """
    logger.info(f"Buscando resultados en: {simod_output_dir}")
    
    # Simod genera resultados en subdirectorios con timestamps
    # Buscamos el directorio más reciente
    if not os.path.exists(simod_output_dir):
        logger.error(f"No se encontró el directorio de salida: {simod_output_dir}")
        return False
    
    # Buscar subdirectorios
    subdirs = [d for d in os.listdir(simod_output_dir) 
               if os.path.isdir(os.path.join(simod_output_dir, d))]
    
    if not subdirs:
        logger.error(f"No se encontraron subdirectorios en: {simod_output_dir}")
        return False
    
    # Obtener el más reciente
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(simod_output_dir, d)))
    best_result_dir = os.path.join(simod_output_dir, latest_subdir, "best_result")
    
    if not os.path.exists(best_result_dir):
        logger.error(f"No se encontró best_result en: {latest_subdir}")
        return False
    
    logger.info(f"Directorio encontrado: {latest_subdir}/best_result")
    
    # Buscar archivos BPMN y JSON
    bpmn_file = os.path.join(best_result_dir, f"{log_name}.bpmn")
    json_file = os.path.join(best_result_dir, f"{log_name}.json")
    
    files_found: list[Tuple[str, str]] = []
    if os.path.exists(bpmn_file):
        files_found.append(("BPMN", bpmn_file))
    else:
        logger.warning(f"No se encontró: {bpmn_file}")
    
    if os.path.exists(json_file):
        files_found.append(("JSON", json_file))
    else:
        logger.warning(f"No se encontró: {json_file}")
    
    if not files_found:
        logger.error("No se encontraron archivos BPMN ni JSON")
        return False
    
    # Copiar archivos a la carpeta objetivo
    logger.info(f"Copiando archivos a: {target_dir}")
    for file_type, source_path in files_found:
        target_path = os.path.join(target_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target_path)
        logger.info(f"{file_type} copiado: {os.path.basename(target_path)}")
    
    return True

def extract_bpmn_json(
    log_path: str,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Función principal: extrae BPMN y JSON desde un log usando Simod.
    
    Args:
        log_path: Ruta al archivo CSV, XES o XES.GZ del log
        config: Diccionario de configuración (si None, se carga desde config.yaml)
    
    Returns:
        True si la extracción fue exitosa, False en caso contrario
    """
    logger.info("=" * 80)
    logger.info("EXTRACCIÓN DE BPMN Y JSON CON SIMOD")
    logger.info("=" * 80)
    
    # Cargar configuración si no se proporciona
    if config is None:
        config = load_config()
        if config is None:
            logger.error("No se pudo cargar la configuración")
            return False
    
    log_config = config.get("log_config", {})
    simod_config = config.get("simod_config", {})
    script_config = config.get("script_config", {})
    
    # Validar que el archivo existe
    if not os.path.exists(log_path):
        logger.error(f"No se encontró el archivo: {log_path}")
        return False
    
    # Obtener información del log
    log_dir = os.path.dirname(os.path.abspath(log_path))
    log_filename = os.path.basename(log_path)
    
    # Detectar formato del log
    log_ext = os.path.splitext(log_filename)[1].lower()
    if log_ext == '.gz':
        # Si es .gz, obtener la extensión del archivo sin comprimir
        log_ext = os.path.splitext(os.path.splitext(log_filename)[0])[1].lower()
    
    is_xes = log_ext in ['.xes']
    log_name = os.path.splitext(log_filename)[0]
    if log_name.endswith('.xes'):
        log_name = os.path.splitext(log_name)[0]
    
    # Determinar directorio de salida
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Si estamos en src/, subir un nivel para llegar a nuevo/
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    output_dir = script_config.get("output_dir")
    if output_dir is None:
        # Por defecto: data/generado-simod dentro de la carpeta "nuevo"
        output_dir = os.path.join(base_dir, "data", "generado-simod")
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Prefijo para directorios temporales
    temp_prefix = script_config.get("temp_dir_prefix", ".simod_temp")
    
    logger.info(f"Log: {log_path}")
    logger.info(f"Directorio del log: {log_dir}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"Nombre: {log_name}")
    
    # Crear directorio temporal para Simod
    temp_input_dir = os.path.join(log_dir, f"{temp_prefix}_input")
    temp_output_dir = os.path.join(log_dir, f"{temp_prefix}_output")
    
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # Paso 1: Preparar el log para Simod
        if is_xes:
            # Simod acepta XES directamente, solo copiar/descomprimir si es necesario
            if log_path.endswith('.gz'):
                # Descomprimir XES.GZ a XES
                xes_path = os.path.join(temp_input_dir, f"{log_name}.xes")
                logger.info(f"Descomprimiendo {log_path} a {xes_path}")
                with gzip.open(log_path, 'rb') as f_in:
                    with open(xes_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logger.info(f"Log descomprimido: {xes_path}")
                log_path_for_simod = xes_path
                simod_log_name = f"{log_name}.xes"
            else:
                # Copiar XES al directorio temporal
                xes_path = os.path.join(temp_input_dir, f"{log_name}.xes")
                shutil.copy2(log_path, xes_path)
                log_path_for_simod = xes_path
                simod_log_name = f"{log_name}.xes"
        else:
            # CSV: comprimir a CSV.GZ
            gz_path = os.path.join(temp_input_dir, f"{log_name}.csv.gz")
            compress_log_to_gz(log_path, gz_path)
            log_path_for_simod = gz_path
            simod_log_name = f"{log_name}.csv.gz"
        
        # Paso 2: Crear archivo de configuración para Simod
        log_format = "xes" if is_xes else "csv"
        config_path = create_config_yaml(log_name, temp_input_dir, simod_config, log_config, log_format)
        config_filename = os.path.basename(config_path)
        
        # Paso 3: Ejecutar Simod
        docker_config = script_config.get("docker", {})
        if not run_simod_docker(temp_input_dir, temp_output_dir, config_filename, docker_config):
            return False
        
        # Paso 4: Copiar resultados a la carpeta de salida
        if not find_and_copy_results(temp_output_dir, output_dir, log_name):
            return False
        
        logger.info("=" * 80)
        logger.info("EXTRACCIÓN COMPLETADA")
        logger.info("=" * 80)
        logger.info(f"Archivos generados en: {output_dir}")
        logger.info(f"  • {log_name}.bpmn")
        logger.info(f"  • {log_name}.json")
        
        return True
        
    finally:
        # Limpiar directorios temporales
        logger.info("Limpiando directorios temporales")
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        logger.info("Limpieza completada")

def main() -> None:
    """Función principal para ejecutar desde línea de comandos"""
    # Cargar configuración
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración")
        sys.exit(1)
    
    log_config = config.get("log_config", {})
    
    # Obtener ruta del log: primero de argumentos, luego de config.yaml
    if len(sys.argv) >= 2:
        log_path = sys.argv[1]
        logger.info(f"Log especificado como argumento: {log_path}")
    elif log_config.get("log_path"):
        log_path = log_config.get("log_path")
        # Si es una ruta relativa, hacerla relativa al directorio base
        if not os.path.isabs(log_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.basename(script_dir) == "src":
                base_dir = os.path.dirname(script_dir)
            else:
                base_dir = script_dir
            log_path = os.path.join(base_dir, log_path)
        logger.info(f"Log leído desde config.yaml: {log_path}")
    else:
        logger.error("No se especificó la ruta del log")
        logger.error("Opciones:")
        logger.error("  1. Como argumento: python extract_bpmn_json.py <ruta_al_log.csv>")
        logger.error("  2. En config.yaml: especificar 'log_config.log_path'")
        sys.exit(1)
    
    if extract_bpmn_json(log_path, config):
        logger.info("¡Proceso completado exitosamente!")
        sys.exit(0)
    else:
        logger.error("El proceso falló")
        sys.exit(1)

if __name__ == "__main__":
    main()

