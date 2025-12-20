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
import time
import threading
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logger_utils import setup_logger

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
    logger.info(f"📦 Comprimiendo {os.path.basename(csv_path)} a {os.path.basename(gz_path)}...")
    start_time = time.time()
    
    # Obtener tamaño original
    original_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    
    with open(csv_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    elapsed = time.time() - start_time
    compressed_size = os.path.getsize(gz_path) / (1024 * 1024)  # MB
    compression_ratio = (1 - compressed_size/original_size) * 100
    
    logger.info(f"✅ Log comprimido: {os.path.basename(gz_path)}")
    logger.info(f"   Tamaño original: {original_size:.2f} MB")
    logger.info(f"   Tamaño comprimido: {compressed_size:.2f} MB")
    logger.info(f"   Compresión: {compression_ratio:.1f}%")
    logger.info(f"   Tiempo: {elapsed:.2f}s")

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
    Ejecuta Simod usando Docker con logging en tiempo real.
    
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
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO SIMOD CON DOCKER")
    logger.info("=" * 80)
    logger.info(f"Imagen Docker: {docker_image}")
    logger.info(f"Directorio de entrada: {input_path}")
    logger.info(f"Directorio de salida: {output_path}")
    logger.info(f"Configuración: {config_file_name}")
    logger.info(f"Comando: {' '.join(docker_command)}")
    logger.info("")
    logger.info("⏳ Simod está ejecutándose... (esto puede tomar varios minutos)")
    logger.info("📊 Mostrando progreso en tiempo real:")
    logger.info("-" * 80)
    
    start_time = time.time()
    last_progress_log = time.time()
    
    # Función para log periódico de progreso
    def log_progress():
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        logger.info(f"⏱️  Tiempo transcurrido: {minutes}m {seconds}s - Simod sigue ejecutándose...")
    
    # Ejecutar con salida en tiempo real
    process = subprocess.Popen(
        docker_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combinar stderr con stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Leer salida línea por línea en tiempo real
    stdout_lines = []
    progress_log_interval = 30  # Log cada 30 segundos si no hay salida
    iteration_count = 0  # Contador de iteraciones para resumen
    last_iteration_log = None
    
    try:
        while True:
            # Verificar si el proceso terminó
            if process.poll() is not None:
                # Leer cualquier salida restante
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        line = line.strip()
                        if line:
                            stdout_lines.append(line)
                            # Log líneas importantes
                            if any(keyword in line.lower() for keyword in [
                                'iteration', 'optimization', 'discovering', 
                                'computing', 'control-flow', 'resource',
                                'error', 'warning', 'traceback', 'failed'
                            ]):
                                logger.info(f"📝 Simod: {line}")
                break
            
            # Leer línea si está disponible (con timeout)
            try:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        stdout_lines.append(line)
                        # Log líneas importantes inmediatamente (filtrado inteligente)
                        line_lower = line.lower()
                        
                        # Filtrar líneas muy verbosas o repetitivas
                        skip_patterns = [
                            'simulation settings:',  # Muy repetitivo (3x por iteración)
                            'prosimossettings(',  # Detalles técnicos innecesarios
                            'posixpath(',  # Rutas internas
                            'finished serializing log',  # Muy frecuente
                            'info:root:running java',  # Detalles técnicos
                        ]
                        
                        if any(pattern in line_lower for pattern in skip_patterns):
                            continue  # Saltar estas líneas
                        
                        # Log líneas importantes
                        important_keywords = [
                            'iteration', 'optimization', 'discovering', 
                            'computing', 'control-flow', 'resource',
                            'error', 'exception', 'traceback', 'failed',
                            'best_result', 'completed', 'finished',
                            'splitminer', 'gateway probabilities',
                            'loss:', 'status:', 'best'
                        ]
                        
                        if any(keyword in line_lower for keyword in important_keywords):
                            # Formatear mejor las líneas importantes
                            if 'control-flow optimization iteration' in line_lower:
                                # Extraer número de iteración
                                import re
                                match = re.search(r'iteration (\d+)', line_lower)
                                if match:
                                    iteration_count = int(match.group(1))
                                    logger.info(f"🔄 Iteración {iteration_count} de optimización de control-flow")
                                    last_iteration_log = time.time()
                            elif 'loss:' in line_lower or 'status:' in line_lower:
                                # Mostrar pérdida y estado de forma más clara
                                if 'loss' in line_lower:
                                    loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                                    status_match = re.search(r"'status':\s*'(\w+)'", line)
                                    if loss_match and status_match:
                                        loss_val = float(loss_match.group(1))
                                        status_val = status_match.group(1)
                                        status_emoji = "✅" if status_val == "ok" else "⚠️"
                                        logger.info(f"📊 {status_emoji} Loss: {loss_val:.6f} | Status: {status_val}")
                                    else:
                                        logger.info(f"📊 Simod: {line[:200]}")
                                else:
                                    logger.info(f"📊 Simod: {line[:200]}")
                            elif 'discovering process model' in line_lower:
                                logger.info(f"🔍 Descubriendo modelo de proceso...")
                            elif 'computing gateway probabilities' in line_lower:
                                logger.info(f"⚙️  Calculando probabilidades de gateways...")
                            elif 'splitminer' in line_lower and 'running' in line_lower:
                                # Extraer epsilon si está disponible
                                epsilon_match = re.search(r"--epsilon['\"]?\s*([\d.]+)", line)
                                if epsilon_match:
                                    epsilon_val = float(epsilon_match.group(1))
                                    logger.info(f"⚙️  SplitMiner ejecutándose (epsilon={epsilon_val:.4f})...")
                                else:
                                    logger.info(f"⚙️  SplitMiner ejecutándose...")
                            else:
                                logger.info(f"📝 Simod: {line[:150]}")
                        last_progress_log = time.time()
                else:
                    # Si no hay salida, log periódico
                    if time.time() - last_progress_log > progress_log_interval:
                        log_progress()
                        last_progress_log = time.time()
            except Exception as e:
                logger.debug(f"Error leyendo salida: {e}")
                time.sleep(0.1)
                continue
            
            time.sleep(0.1)  # Pequeña pausa para no consumir CPU
    
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupción detectada, terminando proceso Docker...")
        process.terminate()
        process.wait()
        return False
    
    # Esperar a que termine completamente
    returncode = process.wait()
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("-" * 80)
    logger.info(f"⏱️  Tiempo total de ejecución: {minutes}m {seconds}s")
    
    if returncode == 0:
        logger.info("✅ Simod ejecutado exitosamente")
        # Mostrar resumen de iteraciones si hay información
        if iteration_count > 0:
            logger.info(f"📊 Resumen: {iteration_count + 1} iteraciones de optimización completadas")
        # Mostrar últimas líneas de salida si hay información relevante
        if stdout_lines:
            relevant_lines = [l for l in stdout_lines[-20:] if any(
                keyword in l.lower() for keyword in [
                    'best_result', 'completed', 'finished', 'success', 'best'
                ]
            )]
            if relevant_lines:
                logger.info("📋 Últimas líneas relevantes:")
                for line in relevant_lines[:5]:  # Solo primeras 5 líneas relevantes
                    logger.info(f"   {line[:200]}")
        return True
    else:
        logger.error("❌ Simod falló")
        logger.error(f"Código de salida: {returncode}")
        # Mostrar últimas líneas de error
        if stdout_lines:
            error_lines = [l for l in stdout_lines if any(
                keyword in l.lower() for keyword in [
                    'error', 'exception', 'traceback', 'failed', 'keyerror'
                ]
            )]
            if error_lines:
                logger.error("📋 Líneas de error encontradas:")
                for line in error_lines[-30:]:  # Últimas 30 líneas de error
                    logger.error(f"   {line}")
            else:
                logger.error("📋 Últimas 50 líneas de salida:")
                for line in stdout_lines[-50:]:
                    logger.error(f"   {line}")
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
    logger.info(f"🔍 Buscando resultados en: {simod_output_dir}")
    
    # Simod genera resultados en subdirectorios con timestamps
    # Buscamos el directorio más reciente
    if not os.path.exists(simod_output_dir):
        logger.error(f"❌ No se encontró el directorio de salida: {simod_output_dir}")
        return False
    
    # Buscar subdirectorios
    subdirs = [d for d in os.listdir(simod_output_dir) 
               if os.path.isdir(os.path.join(simod_output_dir, d))]
    
    if not subdirs:
        logger.error(f"❌ No se encontraron subdirectorios en: {simod_output_dir}")
        logger.info(f"   Contenido del directorio: {os.listdir(simod_output_dir)}")
        return False
    
    logger.info(f"   Encontrados {len(subdirs)} subdirectorio(s)")
    
    # Obtener el más reciente
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(simod_output_dir, d)))
    best_result_dir = os.path.join(simod_output_dir, latest_subdir, "best_result")
    
    logger.info(f"   Directorio más reciente: {latest_subdir}")
    
    if not os.path.exists(best_result_dir):
        logger.error(f"❌ No se encontró best_result en: {latest_subdir}")
        logger.info(f"   Contenido de {latest_subdir}: {os.listdir(os.path.join(simod_output_dir, latest_subdir))}")
        return False
    
    logger.info(f"✅ Directorio encontrado: {latest_subdir}/best_result")
    
    # Buscar archivos BPMN y JSON
    bpmn_file = os.path.join(best_result_dir, f"{log_name}.bpmn")
    json_file = os.path.join(best_result_dir, f"{log_name}.json")
    
    files_found: list[Tuple[str, str]] = []
    if os.path.exists(bpmn_file):
        bpmn_size = os.path.getsize(bpmn_file) / 1024  # KB
        files_found.append(("BPMN", bpmn_file))
        logger.info(f"   ✅ BPMN encontrado: {os.path.basename(bpmn_file)} ({bpmn_size:.2f} KB)")
    else:
        logger.warning(f"   ⚠️  No se encontró: {os.path.basename(bpmn_file)}")
        logger.info(f"      Archivos en best_result: {os.listdir(best_result_dir)}")
    
    if os.path.exists(json_file):
        json_size = os.path.getsize(json_file) / (1024 * 1024)  # MB
        files_found.append(("JSON", json_file))
        logger.info(f"   ✅ JSON encontrado: {os.path.basename(json_file)} ({json_size:.2f} MB)")
    else:
        logger.warning(f"   ⚠️  No se encontró: {os.path.basename(json_file)}")
        logger.info(f"      Archivos en best_result: {os.listdir(best_result_dir)}")
    
    if not files_found:
        logger.error("❌ No se encontraron archivos BPMN ni JSON")
        return False
    
    # Copiar archivos a la carpeta objetivo
    logger.info(f"📋 Copiando archivos a: {target_dir}")
    for file_type, source_path in files_found:
        target_path = os.path.join(target_dir, os.path.basename(source_path))
        start_time = time.time()
        shutil.copy2(source_path, target_path)
        elapsed = time.time() - start_time
        file_size = os.path.getsize(target_path) / (1024 * 1024)  # MB
        logger.info(f"   ✅ {file_type} copiado: {os.path.basename(target_path)} ({file_size:.2f} MB, {elapsed:.2f}s)")
    
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
    log_name = get_log_name_from_path(log_path)
    
    # Determinar directorio de salida (incluyendo nombre del log)
    output_dir_base = script_config.get("output_dir")
    output_dir = build_output_path(output_dir_base, log_name, "simod", default_base="data")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar si los archivos BPMN y JSON ya existen
    bpmn_path = os.path.join(output_dir, f"{log_name}.bpmn")
    json_path = os.path.join(output_dir, f"{log_name}.json")
    
    bpmn_exists = os.path.exists(bpmn_path)
    json_exists = os.path.exists(json_path)
    
    if bpmn_exists and json_exists:
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ARCHIVOS DE SIMOD YA EXISTEN - OMITIENDO EJECUCIÓN")
        logger.info("=" * 80)
        logger.info(f"📁 Directorio: {output_dir}")
        if bpmn_exists:
            bpmn_size = os.path.getsize(bpmn_path) / 1024  # KB
            logger.info(f"  ✅ {log_name}.bpmn ({bpmn_size:.2f} KB)")
        if json_exists:
            json_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
            logger.info(f"  ✅ {log_name}.json ({json_size:.2f} MB)")
        logger.info("")
        logger.info("💡 Para regenerar los archivos, elimínelos primero o use --force")
        logger.info("")
        return True
    elif bpmn_exists or json_exists:
        # Si solo existe uno de los dos, advertir pero continuar
        logger.warning("⚠️  Solo se encontró uno de los archivos:")
        if bpmn_exists:
            logger.warning(f"  ✅ {log_name}.bpmn existe")
        else:
            logger.warning(f"  ❌ {log_name}.bpmn NO existe")
        if json_exists:
            logger.warning(f"  ✅ {log_name}.json existe")
        else:
            logger.warning(f"  ❌ {log_name}.json NO existe")
        logger.warning("  Continuando con la ejecución de Simod para generar los archivos faltantes...")
        logger.info("")
    
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
        logger.info("")
        logger.info("📋 PASO 1: Preparando log para Simod")
        logger.info("-" * 80)
        
        if is_xes:
            # Simod acepta XES directamente, solo copiar/descomprimir si es necesario
            if log_path.endswith('.gz'):
                # Descomprimir XES.GZ a XES
                xes_path = os.path.join(temp_input_dir, f"{log_name}.xes")
                logger.info(f"📦 Descomprimiendo XES.GZ: {os.path.basename(log_path)}")
                start_time = time.time()
                with gzip.open(log_path, 'rb') as f_in:
                    with open(xes_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                elapsed = time.time() - start_time
                xes_size = os.path.getsize(xes_path) / (1024 * 1024)  # MB
                logger.info(f"✅ Log descomprimido: {os.path.basename(xes_path)} ({xes_size:.2f} MB, {elapsed:.2f}s)")
                log_path_for_simod = xes_path
                simod_log_name = f"{log_name}.xes"
            else:
                # Copiar XES al directorio temporal
                xes_path = os.path.join(temp_input_dir, f"{log_name}.xes")
                logger.info(f"📋 Copiando XES al directorio temporal...")
                shutil.copy2(log_path, xes_path)
                xes_size = os.path.getsize(xes_path) / (1024 * 1024)  # MB
                logger.info(f"✅ XES copiado: {os.path.basename(xes_path)} ({xes_size:.2f} MB)")
                log_path_for_simod = xes_path
                simod_log_name = f"{log_name}.xes"
        else:
            # CSV: Preprocesar si es necesario (limpiar espacios en blanco y crear start_time si solo hay time:timestamp)
            column_mapping = log_config.get("column_mapping", {})
            start_time_col = column_mapping.get("start_time", "")
            end_time_col = column_mapping.get("end_time", "")
            
            # Determinar si necesita preprocesamiento
            needs_preprocessing = False
            preprocess_reasons = []
            
            # Verificar si start_time y end_time apuntan al mismo campo
            if start_time_col == end_time_col and start_time_col:
                needs_preprocessing = True
                preprocess_reasons.append(f"crear columna start_time desde {start_time_col}")
            
            # Siempre limpiar espacios en blanco para evitar KeyError en Simod (ej: 'W_Shortened completion ' con espacio)
            needs_preprocessing = True
            preprocess_reasons.append("limpiar espacios en blanco de columnas de texto")
            
            if needs_preprocessing:
                logger.info(f"🔧 Preprocesando CSV: {', '.join(preprocess_reasons)}")
                import pandas as pd
                logger.info(f"   Leyendo CSV: {os.path.basename(log_path)}")
                start_time = time.time()
                df = pd.read_csv(log_path, low_memory=False)
                read_time = time.time() - start_time
                logger.info(f"   CSV leído: {len(df):,} filas, {read_time:.2f}s")
                
                # Limpiar espacios en blanco de columnas de texto relevantes
                text_columns = ['concept:name', 'org:resource', 'case:concept:name']
                for col in text_columns:
                    if col in df.columns:
                        # Contar valores con espacios antes de limpiar
                        trailing_count = (df[col].astype(str).str.endswith(' ').sum())
                        leading_count = (df[col].astype(str).str.startswith(' ').sum())
                        # Limpiar espacios al inicio y al final
                        df[col] = df[col].astype(str).str.strip()
                        total_cleaned = trailing_count + leading_count
                        if total_cleaned > 0:
                            logger.info(f"   Limpiados espacios en '{col}': {total_cleaned} valores corregidos")
                
                # Crear start_time si no existe y es igual a end_time
                if start_time_col in df.columns and 'start_time' not in df.columns:
                    # Crear una nueva columna start_time con el mismo valor que time:timestamp
                    df['start_time'] = df[start_time_col].copy()
                
                # Guardar CSV preprocesado
                preprocessed_csv = os.path.join(temp_input_dir, f"{log_name}_preprocessed.csv")
                logger.info(f"   Guardando CSV preprocesado...")
                df.to_csv(preprocessed_csv, index=False)
                log_path = preprocessed_csv
                
                # Actualizar el mapeo para que Simod use la nueva columna start_time
                if start_time_col == end_time_col and start_time_col:
                    log_config = log_config.copy()
                    log_config['column_mapping'] = column_mapping.copy()
                    log_config['column_mapping']['start_time'] = 'start_time'
                
                csv_size = os.path.getsize(preprocessed_csv) / (1024 * 1024)  # MB
                logger.info(f"✅ CSV preprocesado guardado: {os.path.basename(preprocessed_csv)} ({csv_size:.2f} MB)")
                if start_time_col == end_time_col:
                    logger.info(f"   Mapeo actualizado: start_time ahora apunta a columna 'start_time'")
            
            # Comprimir a CSV.GZ
            logger.info("")
            gz_path = os.path.join(temp_input_dir, f"{log_name}.csv.gz")
            compress_log_to_gz(log_path, gz_path)
            log_path_for_simod = gz_path
            simod_log_name = f"{log_name}.csv.gz"
        
        # Paso 2: Crear archivo de configuración para Simod
        logger.info("")
        logger.info("📋 PASO 2: Creando archivo de configuración para Simod")
        logger.info("-" * 80)
        # Nota: log_config puede haber sido modificado si se preprocesó el CSV
        log_format = "xes" if is_xes else "csv"
        config_path = create_config_yaml(log_name, temp_input_dir, simod_config, log_config, log_format)
        config_filename = os.path.basename(config_path)
        logger.info(f"✅ Configuración creada: {config_filename}")
        
        # Paso 3: Ejecutar Simod
        logger.info("")
        docker_config = script_config.get("docker", {})
        if not run_simod_docker(temp_input_dir, temp_output_dir, config_filename, docker_config):
            return False
        
        # Paso 4: Copiar resultados a la carpeta de salida
        logger.info("")
        logger.info("📋 PASO 4: Buscando y copiando resultados")
        logger.info("-" * 80)
        if not find_and_copy_results(temp_output_dir, output_dir, log_name):
            return False
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ EXTRACCIÓN COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"📁 Archivos generados en: {output_dir}")
        bpmn_path = os.path.join(output_dir, f"{log_name}.bpmn")
        json_path = os.path.join(output_dir, f"{log_name}.json")
        if os.path.exists(bpmn_path):
            bpmn_size = os.path.getsize(bpmn_path) / 1024  # KB
            logger.info(f"  ✅ {log_name}.bpmn ({bpmn_size:.2f} KB)")
        if os.path.exists(json_path):
            json_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
            logger.info(f"  ✅ {log_name}.json ({json_size:.2f} MB)")
        logger.info("")
        
        return True
        
    finally:
        # Limpiar directorios temporales
        logger.info("")
        logger.info("🧹 Limpiando directorios temporales...")
        if os.path.exists(temp_input_dir):
            logger.info(f"   Eliminando: {temp_input_dir}")
            shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir):
            logger.info(f"   Eliminando: {temp_output_dir}")
            shutil.rmtree(temp_output_dir)
        logger.info("✅ Limpieza completada")

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
        # Si es una ruta relativa, hacerla relativa al directorio base del proyecto
        if not os.path.isabs(log_path):
            # Encontrar el directorio raíz del proyecto
            # Este script está en src/causal-gym/, así que subimos dos niveles
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # script_dir = src/causal-gym/
            src_dir = os.path.dirname(script_dir)  # src/
            project_root = os.path.dirname(src_dir)  # proyecto raíz
            log_path = os.path.join(project_root, log_path)
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

