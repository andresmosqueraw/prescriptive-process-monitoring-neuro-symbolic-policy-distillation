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

# ==========================================
# 0. SETUP & MOCK IMPORTS
# ==========================================
# Intentamos ubicar Prosimos local y agregarlo al sys.path.
#
# En este repo, Prosimos suele estar en:
#   paper1/repos-asis-online-predictivo/.../libraries-used/Prosimos/
# y dentro existe el paquete/namespace `prosimos/`.

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
        return None
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def _maybe_add_local_prosimos_to_syspath() -> None:
    config = load_config()
    if config is None:
        print("❌ No se pudo cargar la configuración desde configs/config.yaml")
        print("   Asegúrate de que el archivo configs/config.yaml existe")
        return
    
    if not config.get("external_repos"):
        print("❌ No se encontró la sección 'external_repos' en configs/config.yaml")
        print("   Agrega la siguiente sección a tu config.yaml:")
        print("   external_repos:")
        print("     prosimos_path: /ruta/a/Prosimos")
        return
    
    prosimos_path = config["external_repos"].get("prosimos_path")
    
    if not prosimos_path:
        print("❌ No se encontró 'prosimos_path' en configs/config.yaml")
        print("   Configura la ruta en configs/config.yaml:")
        print("   external_repos:")
        print("     prosimos_path: /ruta/a/Prosimos")
        return
    
    if not os.path.exists(prosimos_path):
        print(f"❌ La ruta configurada no existe: {prosimos_path}")
        print("   Verifica que la ruta en configs/config.yaml sea correcta")
        return
    
    if not os.path.isdir(os.path.join(prosimos_path, "prosimos")):
        print(f"❌ La ruta no contiene el directorio 'prosimos': {prosimos_path}")
        print("   Verifica que la ruta apunte al directorio raíz de Prosimos")
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
    print("⚠️  ADVERTENCIA: Librería 'prosimos' no detectada.")
    print(f"   Motivo: {e!r}")
    print("   Se usarán clases Mock para demostrar la arquitectura.")
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
        print("   [Sim] Running simulation episode...")
        return None

# ==========================================
# 1. COMPONENTES DE LA ARQUITECTURA
# ==========================================

class SymbolicSafetyGuard:
    """
    Capa de Seguridad Simbólica.
    Verifica si una acción viola reglas de negocio o restricciones de recursos (LTL).
    """
    def __init__(self, rules_config):
        self.rules = rules_config
        self.resource_budget = 1000 # Ejemplo de restricción de recursos
        
    def is_safe(self, case_state, proposed_action):
        """
        Retorna True si la acción es segura y cumple el presupuesto.
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
        
    def calculate_reward(self, action, outcome_success, propensity_score=0.5):
        """
        Calcula la recompensa ajustada causalmente.
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
    def __init__(self, action_space=None):
        self.q_table = {} # State -> {Action -> Value}
        self.epsilon = 0.1 # Exploración
        self.action_space = action_space
        
    def get_action(self, state_str, possible_actions=None):
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

    def update(self, state, action, reward, next_state):
        # Q-Learning Update Rule (Simplificada)
        alpha = 0.1
        gamma = 0.9
        
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

# Buffer global para guardar experiencias para la Fase 3
EXPERIENCE_BUFFER = [] # Lista de dicts: {state, action, reward, next_state}

class NeuroSymbolicFlowSelector:
    """
    Esta clase REEMPLAZA al OutgoingFlowSelector original de Prosimos.
    Intercepta las decisiones de XOR Gateway.
    """
    def __init__(self, original_choose_fn, agent, safety_guard, reward_engine):
        self.original_choose_fn = original_choose_fn # Mantener lógica original como fallback
        self.agent = agent
        self.safety = safety_guard
        self.reward_engine = reward_engine
        
    def choose_outgoing_flow(self, e_info, element_probability, all_attributes, gateway_conditions):
        """
        Método inyectado que se ejecuta en cada decisión del simulador.
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
            "state_feature_vector": current_state, # Feature vector real iría aquí
            "action_taken": final_action,
            "reward_causal": reward,
            "was_safe": is_safe,
            "next_state": "terminal" # Simplificado
        }
        EXPERIENCE_BUFFER.append(experience)
        
        # 7. Actualizar Agente (Online Learning)
        self.agent.update(current_state, final_action, reward, "terminal")
        
        return [(final_action, None)]

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================

def run_causal_gym_training(bpmn_path, json_path, episodes=10):
    print("="*80)
    print("🏋️‍♂️  CAUSAL-GYM: INICIANDO ENTRENAMIENTO NEURO-SIMBÓLICO")
    print("="*80)
    
    # 1. Configuración de Componentes
    agent = RLAgent(action_space=None)
    safety = SymbolicSafetyGuard(rules_config={})
    reward_engine = CausalRewardEngine()
    
    # 2. Preparar Monkey Patching
    print("\n💉 Inyectando NeuroSymbolicFlowSelector en el runtime de Prosimos...")
    
    original_choose = OutgoingFlowSelector.choose_outgoing_flow
    neuro_selector = NeuroSymbolicFlowSelector(original_choose, agent, safety, reward_engine)

    def patched_choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions):
        return neuro_selector.choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions)

    # Aplicar el parche (mantener staticmethod)
    OutgoingFlowSelector.choose_outgoing_flow = staticmethod(patched_choose_outgoing_flow)
    print("✅ Parche aplicado exitosamente.")

    # 3. Bucle de Entrenamiento (Episodios)
    for episode in range(1, episodes + 1):
        print(f"\n🎬 Episodio {episode}/{episodes}")
        
        # Reiniciar presupuesto por episodio
        safety.resource_budget = 100 
        
        # Ejecutar Simulación (Prosimos)
        # Prosimos usará nuestro selector parcheado internamente en los gateways
        run_simulation(bpmn_path, json_path, total_cases=50)
        
        print(f"   Buffer size: {len(EXPERIENCE_BUFFER)} experiencias recolectadas")

    # 4. Exportar Experience Buffer para Fase 3 (Distillation)
    output_file = os.path.join(os.getcwd(), "data", "generado-rl-train/experience_buffer.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if EXPERIENCE_BUFFER:
        keys = EXPERIENCE_BUFFER[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(EXPERIENCE_BUFFER)
        print(f"\n💾 Experience Buffer guardado en: {output_file}")
        print("   Listo para Phase 3: Policy Distillation")
    else:
        print("\n⚠️  No se recolectaron experiencias.")

if __name__ == "__main__":
    # Rutas default (ajustar según tu estructura de carpetas)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Si estamos en src/, subir un nivel para llegar a nuevo/
    if os.path.basename(script_dir) == "src":
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir

    bpmn_file = os.path.join(base_dir, "data/generado-simod/PurchasingExample.bpmn")
    json_file = os.path.join(base_dir, "data/generado-simod/PurchasingExample.json")
    
    # Asegurar que el CSV se guarde bajo nuevo/data aunque el CWD sea otro
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.chdir(base_dir)

    run_causal_gym_training(bpmn_file, json_file, episodes=5)