#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2: CAUSAL-GYM TRAINER
---------------------------
Este script implementa el entorno de entrenamiento "Causal-Gym".
Realiza lo siguiente:
1. Carga la configuración y los modelos AS-IS.
2. Inicializa un Agente de RL (Q-Learning simplificado o Proxy para PPO).
3. "Monkey-Patches" (parcha en tiempo de ejecución) el simulador PROSIMOS.
4. Ejecuta bucles de simulación donde el Agente toma decisiones.
5. Aplica 'Symbolic Guards' (Reglas) y 'Causal Rewards' (IPW).
6. Guarda el 'Experience Buffer' para la Fase 3 (Distillation).
"""

import os
import sys
import csv
import random
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, List, Callable, Tuple

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logging import setup_logger

# Configurar logger
logger = setup_logger(__name__)

# ==========================================
# 0. SETUP & MOCK IMPORTS
# ==========================================
# Intentamos ubicar Prosimos local y agregarlo al sys.path.
#
# En este repo, Prosimos suele estar en:
#   paper1/repos-asis-online-predictivo/.../libraries-used/Prosimos/
# y dentro existe el paquete/namespace `prosimos/`.

# load_config ahora se importa de utils.config

def _maybe_add_local_prosimos_to_syspath() -> None:
    """Agrega Prosimos al sys.path desde la configuración."""
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración desde configs/config.yaml")
        logger.error("Asegúrate de que el archivo configs/config.yaml existe")
        return
    
    if not config.get("external_repos"):
        logger.error("No se encontró la sección 'external_repos' en configs/config.yaml")
        logger.error("Agrega la siguiente sección a tu config.yaml:")
        logger.error("  external_repos:")
        logger.error("    prosimos_path: /ruta/a/Prosimos")
        return
    
    prosimos_path = config["external_repos"].get("prosimos_path")
    
    if not prosimos_path:
        logger.error("No se encontró 'prosimos_path' en configs/config.yaml")
        logger.error("Configura la ruta en configs/config.yaml:")
        logger.error("  external_repos:")
        logger.error("    prosimos_path: /ruta/a/Prosimos")
        return
    
    if not os.path.exists(prosimos_path):
        logger.error(f"La ruta configurada no existe: {prosimos_path}")
        logger.error("Verifica que la ruta en configs/config.yaml sea correcta")
        return
    
    if not os.path.isdir(os.path.join(prosimos_path, "prosimos")):
        logger.error(f"La ruta no contiene el directorio 'prosimos': {prosimos_path}")
        logger.error("Verifica que la ruta apunte al directorio raíz de Prosimos")
        return
    
    if prosimos_path not in sys.path:
        sys.path.insert(0, prosimos_path)


_maybe_add_local_prosimos_to_syspath()

# Intentamos importar Prosimos. Si no está, usamos Mocks para que el script no rompa.
try:
    # IMPORT ORDER IMPORTANTE:
    # - Si importamos outgoing_flow_selector primero puede ocurrir un ImportError circular.
    import prosimos.simulation_engine as prosimos_sim_engine
    from prosimos.outgoing_flow_selector import OutgoingFlowSelector
    from prosimos.control_flow_manager import BPMN

    run_simulation = prosimos_sim_engine.run_simulation
    PROSIMOS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Librería 'prosimos' no detectada. Motivo: {e!r}")
    logger.warning("Se usarán clases Mock para demostrar la arquitectura.")
    PROSIMOS_AVAILABLE = False
    
    # Mock classes/functions to allow script execution/demonstration
    class OutgoingFlowSelector:
        @staticmethod
        def choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions):
            # elige una salida al azar (si aplica)
            flows = getattr(e_info, "outgoing_flows", []) or []
            if len(flows) <= 1:
                return [(flows[0], None)] if flows else []
            return [(random.choice(list(flows)), None)]

    class _MockBPMN:
        EXCLUSIVE_GATEWAY = object()

    BPMN = _MockBPMN

    def run_simulation(*args, **kwargs):
        logger.debug("[Sim] Running simulation episode...")
        return None

# ==========================================
# 1. COMPONENTES DE LA ARQUITECTURA
# ==========================================

class SymbolicSafetyGuard:
    """
    Capa de Seguridad Simbólica.
    Verifica si una acción viola reglas de negocio o restricciones de recursos (LTL).
    """
    def __init__(self, rules_config: Dict[str, Any], resource_budget: int = 1000):
        """
        Inicializa el guard de seguridad simbólica.
        
        Args:
            rules_config: Configuración de reglas
            resource_budget: Presupuesto inicial de recursos
        """
        self.rules = rules_config
        self.resource_budget = resource_budget
        
    def is_safe(self, case_state: Dict[str, Any], proposed_action: str) -> bool:
        """
        Retorna True si la acción es segura y cumple el presupuesto.
        
        Args:
            case_state: Estado actual del caso
            proposed_action: Acción propuesta
        
        Returns:
            True si la acción es segura, False en caso contrario
        """
        # Regla 1: Restricción de Presupuesto
        if self.resource_budget <= 0:
            if proposed_action == "Intervention_Call":
                return False # Bloqueado por falta de presupuesto
        
        # Regla 2: Lógica de Negocio (Ejemplo LTL: Si riesgo alto, no ignorar)
        if case_state.get("risk_score", 0) > 0.8 and proposed_action == "Ignore":
            return False # Bloqueado por regla de seguridad
            
        return True

    def consume_budget(self, action):
        if action == "Intervention_Call":
            self.resource_budget -= 10

class CausalRewardEngine:
    """
    Motor de Recompensas Causales.
    Usa IPW (Inverse Propensity Weighting) para corregir el sesgo.
    R = Revenue - Cost + CausalCorrection
    """
    def __init__(self):
        self.intervention_cost = 20
        self.success_reward = 100
        
    def calculate_reward(
        self,
        action: str,
        outcome_success: bool,
        propensity_score: float = 0.5
    ) -> float:
        """
        Calcula la recompensa ajustada causalmente.
        
        Args:
            action: Acción tomada
            outcome_success: Si el resultado fue exitoso
            propensity_score: Score de propensión para IPW
        
        Returns:
            Recompensa calculada
        """
        base_reward = 0
        
        # Costo inmediato de la acción
        if action == "Intervention_Call":
            base_reward -= self.intervention_cost
            
        # Recompensa por resultado (IPW Adjustment)
        # En entrenamiento real, esto se ajusta comparando con el contrafactual
        if outcome_success:
            # Si tuvo éxito, premiamos inversamente a la probabilidad de haber sido tratado
            # Esto da más valor a casos "difíciles" que tuvieron éxito
            ipw_factor = 1.0 / propensity_score 
            base_reward += (self.success_reward * ipw_factor)
        else:
            base_reward -= 5 # Penalización por retraso/fallo
            
        return base_reward

class RLAgent:
    """
    Agente simple de RL (Q-Learning) para demostración.
    En producción, esto sería un wrapper para Stable-Baselines3 (PPO/DQN).
    """
    def __init__(
        self,
        action_space: Optional[List[str]] = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1
    ):
        """
        Inicializa el agente RL.
        
        Args:
            action_space: Espacio de acciones disponible
            learning_rate: Tasa de aprendizaje
            epsilon: Probabilidad de exploración (epsilon-greedy)
        """
        self.q_table: Dict[str, Dict[str, float]] = {}  # State -> {Action -> Value}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.action_space = action_space
        
    def get_action(self, state_str: str, possible_actions: Optional[List[str]] = None) -> str:
        """
        Selecciona una acción usando epsilon-greedy.
        
        Args:
            state_str: Representación del estado como string
            possible_actions: Lista de acciones posibles (si None, usa self.action_space)
        
        Returns:
            Acción seleccionada
        
        Raises:
            ValueError: Si no hay acciones disponibles
        """
        possible_actions = list(possible_actions) if possible_actions is not None else self.action_space
        if not possible_actions:
            raise ValueError("No hay acciones disponibles para seleccionar")

        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        if state_str not in self.q_table:
            self.q_table[state_str] = {a: 0.0 for a in possible_actions}
        else:
            # agregar acciones nuevas (si cambia el set de acciones posibles)
            for a in possible_actions:
                self.q_table[state_str].setdefault(a, 0.0)
        
        # Retornar acción con mayor Q-value
        return max(self.q_table[state_str], key=self.q_table[state_str].get)

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        gamma: float = 0.9
    ) -> None:
        """
        Actualiza la Q-table usando Q-Learning.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            gamma: Factor de descuento
        """
        alpha = self.learning_rate
        
        # Inicialización robusta (action_space puede ser dinámica)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state].setdefault(action, 0.0)

        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        
        old_val = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_val = old_val + alpha * (reward + gamma * next_max - old_val)
        self.q_table[state][action] = new_val

# ==========================================
# 2. THE MONKEY PATCH (INYECCIÓN DE CÓDIGO)
# ==========================================

class ExperienceBuffer:
    """
    Buffer para almacenar experiencias del agente RL.
    Encapsula el estado que antes era global.
    """
    def __init__(self):
        """Inicializa el buffer vacío."""
        self.buffer: List[Dict[str, Any]] = []
    
    def append(self, experience: Dict[str, Any]) -> None:
        """
        Agrega una experiencia al buffer.
        
        Args:
            experience: Diccionario con la experiencia (state, action, reward, etc.)
        """
        self.buffer.append(experience)
    
    def __len__(self) -> int:
        """Retorna el número de experiencias en el buffer."""
        return len(self.buffer)
    
    def __iter__(self):
        """Permite iterar sobre las experiencias."""
        return iter(self.buffer)
    
    def clear(self) -> None:
        """Limpia el buffer."""
        self.buffer.clear()

class NeuroSymbolicFlowSelector:
    """
    Esta clase REEMPLAZA al OutgoingFlowSelector original de Prosimos.
    Intercepta las decisiones de XOR Gateway.
    """
    def __init__(
        self,
        original_choose_fn: Callable,
        agent: RLAgent,
        safety_guard: SymbolicSafetyGuard,
        reward_engine: CausalRewardEngine,
        experience_buffer: ExperienceBuffer
    ):
        """
        Inicializa el selector neuro-simbólico.
        
        Args:
            original_choose_fn: Función original de Prosimos como fallback
            agent: Agente RL
            safety_guard: Guard de seguridad simbólica
            reward_engine: Motor de recompensas causales
            experience_buffer: Buffer para almacenar experiencias
        """
        self.original_choose_fn = original_choose_fn
        self.agent = agent
        self.safety = safety_guard
        self.reward_engine = reward_engine
        self.experience_buffer = experience_buffer
        
    def choose_outgoing_flow(
        self,
        e_info: Any,
        element_probability: Any,
        all_attributes: Any,
        gateway_conditions: Any
    ) -> List[Tuple[Any, Optional[Any]]]:
        """
        Método inyectado que se ejecuta en cada decisión del simulador.
        
        Args:
            e_info: Información del elemento
            element_probability: Probabilidades del elemento
            all_attributes: Todos los atributos
            gateway_conditions: Condiciones del gateway
        
        Returns:
            Lista de tuplas (flow, condition) seleccionadas
        """
        # Si no es XOR, usar la lógica original de Prosimos
        try:
            if not (getattr(e_info, "type", None) is BPMN.EXCLUSIVE_GATEWAY and len(getattr(e_info, "outgoing_flows", []) or []) > 1):
                return self.original_choose_fn(e_info, element_probability, all_attributes, gateway_conditions)
        except Exception:
            # si algo raro pasa, fallback a original
            return self.original_choose_fn(e_info, element_probability, all_attributes, gateway_conditions)

        possible_paths = list(getattr(e_info, "outgoing_flows", []) or [])

        # 1. Estado (simplificado): gateway_id + algunas features de atributos
        current_state = f"gateway={getattr(e_info, 'id', 'unknown')}"

        # 2. El Agente RL propone una acción (una de las salidas)
        proposed_path = self.agent.get_action(current_state, possible_actions=possible_paths)

        # 3. Symbolic Safety Check (Compliance & Resources)
        risk_score = 0.9
        if isinstance(all_attributes, dict):
            risk_score = float(all_attributes.get("risk_score", risk_score))
        is_safe = self.safety.is_safe({"risk_score": risk_score}, proposed_path)
        
        final_action = proposed_path
        if not is_safe:
            # Si es inseguro, forzamos la ruta segura (Fallback Policy)
            # Asumimos que la ruta segura es la alternativa
            alternatives = [p for p in possible_paths if p != proposed_path]
            final_action = alternatives[0] if alternatives else proposed_path
            
        # 4. Consumir Presupuesto (Simulación)
        self.safety.consume_budget(final_action)
        
        # 5. Calcular Recompensa (Mock Outcome)
        # En simulación real, esperamos al siguiente evento para saber el outcome.
        # Aquí simulamos un resultado inmediato para el ejemplo.
        simulated_outcome = random.random() > 0.3 # 70% éxito
        reward = self.reward_engine.calculate_reward(final_action, simulated_outcome)
        
        # 6. Guardar en Experience Buffer (Para Fase 3 - Distillation)
        experience = {
            "case_id": None,
            "timestamp": datetime.now().isoformat(),
            "state_feature_vector": current_state,  # Feature vector real iría aquí
            "action_taken": final_action,
            "reward_causal": reward,
            "was_safe": is_safe,
            "next_state": "terminal"  # Simplificado
        }
        self.experience_buffer.append(experience)
        
        # 7. Actualizar Agente (Online Learning)
        self.agent.update(current_state, final_action, reward, "terminal")
        
        return [(final_action, None)]

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================

def run_causal_gym_training(
    bpmn_path: str,
    json_path: str,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Función principal de entrenamiento RL.
    
    Args:
        bpmn_path: Ruta al archivo BPMN
        json_path: Ruta al archivo JSON de parámetros
        config: Diccionario de configuración (si None, se carga desde config.yaml)
    
    Returns:
        True si el entrenamiento fue exitoso, False en caso contrario
    """
    # Cargar configuración si no se proporciona
    if config is None:
        config = load_config()
        if config is None:
            logger.error("No se pudo cargar la configuración desde configs/config.yaml")
            return False
    
    rl_config = config.get("rl_config", {})
    script_config = config.get("script_config", {})
    
    # Obtener parámetros de configuración
    episodes = rl_config.get("episodes", 10)
    total_cases = rl_config.get("total_cases", 50)
    learning_rate = rl_config.get("learning_rate", 0.01)
    epsilon = rl_config.get("epsilon", 0.1)
    resource_budget = rl_config.get("resource_budget", 100)
    
    logger.info("="*80)
    logger.info("CAUSAL-GYM: INICIANDO ENTRENAMIENTO NEURO-SIMBÓLICO")
    logger.info("="*80)
    logger.info(f"Configuración:")
    logger.info(f"  • Episodios: {episodes}")
    logger.info(f"  • Casos por episodio: {total_cases}")
    logger.info(f"  • Learning rate: {learning_rate}")
    logger.info(f"  • Epsilon (exploración): {epsilon}")
    logger.info(f"  • Presupuesto de recursos: {resource_budget}")
    
    # 1. Configuración de Componentes
    agent = RLAgent(action_space=None, learning_rate=learning_rate, epsilon=epsilon)
    
    safety = SymbolicSafetyGuard(rules_config={}, resource_budget=resource_budget)
    
    reward_engine = CausalRewardEngine()
    
    # Crear buffer de experiencias (eliminando estado global)
    experience_buffer = ExperienceBuffer()
    
    # 2. Preparar Monkey Patching
    logger.info("Inyectando NeuroSymbolicFlowSelector en el runtime de Prosimos...")
    
    original_choose = OutgoingFlowSelector.choose_outgoing_flow
    neuro_selector = NeuroSymbolicFlowSelector(
        original_choose, agent, safety, reward_engine, experience_buffer
    )

    def patched_choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions):
        return neuro_selector.choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions)

    # Aplicar el parche (mantener staticmethod)
    OutgoingFlowSelector.choose_outgoing_flow = staticmethod(patched_choose_outgoing_flow)
    logger.info("Parche aplicado exitosamente.")

    # 3. Bucle de Entrenamiento (Episodios)
    for episode in range(1, episodes + 1):
        logger.info(f"Episodio {episode}/{episodes}")
        
        # Reiniciar presupuesto por episodio
        safety.resource_budget = resource_budget
        
        # Ejecutar Simulación (Prosimos)
        # Prosimos usará nuestro selector parcheado internamente en los gateways
        run_simulation(bpmn_path, json_path, total_cases=total_cases)
        
        logger.info(f"  Buffer size: {len(experience_buffer)} experiencias recolectadas")

    # 4. Exportar Experience Buffer para Fase 3 (Distillation)
    # Obtener nombre del log desde config para construir rutas
    log_config = config.get("log_config", {})
    log_path_for_name = log_config.get("log_path")
    if log_path_for_name:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == "src":
            base_dir = os.path.dirname(script_dir)
        else:
            base_dir = script_dir
        if not os.path.isabs(log_path_for_name):
            log_path_for_name = os.path.join(base_dir, log_path_for_name)
        log_name = get_log_name_from_path(log_path_for_name)
    else:
        # Fallback: usar nombre genérico
        log_name = "default"
    
    # Obtener directorio de salida para experience buffer (incluyendo nombre del log)
    rl_output_dir_base = script_config.get("rl_output_dir")
    rl_output_dir = build_output_path(rl_output_dir_base, log_name, "rl", default_base="data")
    
    os.makedirs(rl_output_dir, exist_ok=True)
    output_file = os.path.join(rl_output_dir, "experience_buffer.csv")
    
    if experience_buffer.buffer:
        keys = experience_buffer.buffer[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(experience_buffer.buffer)
        logger.info(f"Experience Buffer guardado en: {output_file}")
        logger.info("Listo para Phase 3: Policy Distillation")
        return True
    else:
        logger.warning("No se recolectaron experiencias.")
        return False

def main() -> None:
    """Función principal para ejecutar desde línea de comandos"""
    # Cargar configuración
    config = load_config()
    if config is None:
        logger.error("No se pudo cargar la configuración desde configs/config.yaml")
        sys.exit(1)
    
    log_config = config.get("log_config", {})
    script_config = config.get("script_config", {})
    
    # Obtener directorio base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    # Obtener ruta del log desde log_path
    log_path = log_config.get("log_path")
    if not log_path:
        logger.error("No se especificó log_path en config.yaml")
        sys.exit(1)
    
    # Si es relativa, hacerla absoluta
    if not os.path.isabs(log_path):
        log_path = os.path.join(base_dir, log_path)
    
    # Obtener nombre del log (sin extensión)
    log_name = get_log_name_from_path(log_path)
    
    # Obtener directorio de salida de Simod (incluyendo nombre del log)
    simod_output_dir_base = script_config.get("output_dir")
    simod_output_dir = build_output_path(simod_output_dir_base, log_name, "simod", default_base="data")
    
    # Construir rutas de BPMN y JSON
    bpmn_file = os.path.join(simod_output_dir, f"{log_name}.bpmn")
    json_file = os.path.join(simod_output_dir, f"{log_name}.json")
    
    # Verificar que existan
    if not os.path.exists(bpmn_file):
        logger.error(f"No se encontró el archivo BPMN: {bpmn_file}")
        logger.error("Ejecuta primero extract_bpmn_json.py para generar los archivos")
        sys.exit(1)
    
    if not os.path.exists(json_file):
        logger.error(f"No se encontró el archivo JSON: {json_file}")
        logger.error("Ejecuta primero extract_bpmn_json.py para generar los archivos")
        sys.exit(1)
    
    # Cambiar al directorio base
    os.chdir(base_dir)
    
    # Ejecutar entrenamiento
    success = run_causal_gym_training(bpmn_file, json_file, config)
    if not success:
        logger.error("El entrenamiento falló")
        sys.exit(1)

if __name__ == "__main__":
    main()