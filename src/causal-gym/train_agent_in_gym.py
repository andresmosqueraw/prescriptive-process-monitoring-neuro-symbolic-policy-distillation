#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 2: CAUSAL-GYM TRAINER
---------------------------
Entrena un agente RL inyectado en Prosimos (Monkey-Patching).
Usa estados parciales, recompensas causales y reglas de seguridad.
"""

import os
import sys
import csv
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils.config import load_config, get_log_name_from_path, build_output_path
from utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# ==========================================
# 0. SETUP & IMPORTS
# ==========================================
def _add_prosimos_path():
    config = load_config()
    if config and config.get("external_repos"):
        path = config["external_repos"].get("prosimos_path")
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

_add_prosimos_path()

try:
    import prosimos.simulation_engine as prosimos_sim_engine
    from prosimos.outgoing_flow_selector import OutgoingFlowSelector
    from prosimos.control_flow_manager import BPMN
    run_simulation = prosimos_sim_engine.run_simulation
    PROSIMOS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Prosimos no encontrado. Usando Mocks.")
    PROSIMOS_AVAILABLE = False
    class OutgoingFlowSelector:
        @staticmethod
        def choose_outgoing_flow(e, p, a, g): return [(getattr(e, "outgoing_flows", [])[0], None)] if getattr(e, "outgoing_flows", []) else []
    class _MockBPMN: EXCLUSIVE_GATEWAY = object()
    BPMN = _MockBPMN
    def run_simulation(*args, **kwargs): pass

# ==========================================
# 1. COMPONENTES NEURO-SIMBÓLICOS
# ==========================================

class SymbolicSafetyGuard:
    """Reglas de negocio duras (Hard Constraints)."""
    def __init__(self, resource_budget=1000):
        self.resource_budget = resource_budget

    def is_safe(self, case_state: Dict, action: str) -> bool:
        # 1. Presupuesto
        if self.resource_budget <= 0 and "Call" in str(action):
            return False
        
        # 2. Regla de Negocio (BPI 2017): No llamar si monto < 5000
        try:
            amount = float(case_state.get("case:RequestedAmount", 0))
        except:
            amount = 0
            
        if "Call" in str(action) and amount < 5000:
            return False
            
        return True

    def consume(self, action):
        if "Call" in str(action):
            self.resource_budget -= 20

class CausalRewardEngine:
    """
    Recompensa con ajuste IPW - MEJORADA para selectividad.
    
    Objetivo: Aprender CUÁNDO intervenir, no siempre intervenir.
    """
    def calculate(self, action: str, success: bool, propensity=0.5, case_state: Dict = None) -> float:
        reward = 0.0
        is_intervention = "Call" in str(action) or "node_" in str(action)
        
        if is_intervention:
            # Costo base de intervención
            reward -= 20.0
            
            if success:
                # Éxito CON intervención: recompensa moderada
                # IPW: Premiar más los éxitos difíciles
                reward += (100.0 / max(propensity, 0.1))
            else:
                # Fracaso CON intervención: penalización extra (desperdicio)
                reward -= 30.0
        else:
            # NO intervención
            if success:
                # Éxito SIN intervención: ¡muy bueno! (ahorro de recursos)
                reward += 80.0  # Premio por no intervenir cuando no era necesario
            else:
                # Fracaso SIN intervención: penalización por no actuar
                # Pero menor que intervenir y fallar (oportunidad perdida)
                reward -= 10.0
        
        # Bonus por selectividad basada en features del caso
        if case_state:
            try:
                amount = float(case_state.get("case:RequestedAmount", 0))
                # Casos con monto alto son más importantes
                if amount > 20000 and is_intervention and success:
                    reward += 50.0  # Bonus por intervenir correctamente en casos importantes
                elif amount < 5000 and not is_intervention:
                    reward += 20.0  # Bonus por NO intervenir en casos pequeños
            except:
                pass
        
        return reward

class RLAgent:
    """Q-Learning simple."""
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state, actions):
        if not actions: return None
        if random.random() < self.epsilon: return random.choice(actions)
        
        if state not in self.q: self.q[state] = {a: 0.0 for a in actions}
        for a in actions:
            if a not in self.q[state]: self.q[state][a] = 0.0
            
        return max(self.q[state], key=self.q[state].get)

    def update(self, state, action, reward, next_state):
        if state not in self.q: self.q[state] = {}
        if next_state not in self.q: self.q[next_state] = {}
        if action not in self.q[state]: self.q[state][action] = 0.0
        
        old = self.q[state][action]
        nxt = max(self.q[next_state].values()) if self.q[next_state] else 0.0
        self.q[state][action] = old + self.alpha * (reward + self.gamma * nxt - old)

class ExperienceBuffer:
    def __init__(self): self.data = []
    def add(self, item): self.data.append(item)

# ==========================================
# 2. MONKEY PATCH
# ==========================================

class NeuroSelector:
    def __init__(self, original, agent, safety, reward, buffer):
        self.original = original
        self.agent = agent
        self.safety = safety
        self.reward = reward
        self.buffer = buffer

    def choose(self, e_info, probs, attrs, conds):
        # Fallback si no es XOR
        try:
            is_xor = getattr(e_info, "type", None) is BPMN.EXCLUSIVE_GATEWAY
            flows = list(getattr(e_info, "outgoing_flows", []) or [])
            if not is_xor or len(flows) <= 1:
                return self.original(e_info, probs, attrs, conds)
        except:
            return self.original(e_info, probs, attrs, conds)

        # --- ESTADO CONTEXTUAL ---
        amt = attrs.get('case:RequestedAmount', 0)
        app_type = attrs.get('case:ApplicationType', 'Unknown')
        state = f"loc={getattr(e_info, 'id')}|amt={amt}|type={app_type}"
        
        # 1. RL Action
        action = self.agent.get_action(state, flows)
        
        # 2. Safety
        is_safe = self.safety.is_safe(attrs, action)
        final_action = action
        if not is_safe:
            alts = [f for f in flows if f != action]
            if alts: final_action = alts[0]

        # 3. Reward & Cost
        self.safety.consume(final_action)
        # Simulación de resultado (Mock para training)
        # La probabilidad de éxito depende del monto y la intervención
        base_success_rate = 0.4
        try:
            amount = float(attrs.get('case:RequestedAmount', 0))
            # Casos con monto alto tienen menor probabilidad base de éxito
            if amount > 20000:
                base_success_rate = 0.25
            elif amount > 10000:
                base_success_rate = 0.35
            # Intervenir mejora la probabilidad de éxito (efecto causal)
            if "Call" in str(final_action) or "node_" in str(final_action):
                # Lift del 20% por intervenir
                base_success_rate = min(0.8, base_success_rate + 0.20)
        except:
            pass
        
        success = random.random() < base_success_rate
        
        r = self.reward.calculate(final_action, success, propensity=0.5, case_state=attrs)

        # 4. Buffer
        self.buffer.add({
            "timestamp": datetime.now().isoformat(),
            "state_feature_vector": state,
            "action_taken": str(final_action),
            "reward_causal": r,
            "was_safe": is_safe
        })

        # 5. Learn
        self.agent.update(state, final_action, r, "terminal")
        
        return [(final_action, None)]

# ==========================================
# 3. EXECUTION
# ==========================================

def train(bpmn, json_path, state_path=None, config=None):
    if not config: config = load_config()
    rl_conf = config.get("rl_config", {})
    
    logger.info("="*60)
    logger.info("🏋️ ENTRENAMIENTO CAUSAL-GYM")
    logger.info("="*60)
    
    agent = RLAgent(epsilon=rl_conf.get("epsilon", 0.1))
    safety = SymbolicSafetyGuard(rl_conf.get("resource_budget", 1000))
    reward = CausalRewardEngine()
    buffer = ExperienceBuffer()
    
    # Patching
    orig = OutgoingFlowSelector.choose_outgoing_flow
    selector = NeuroSelector(orig, agent, safety, reward, buffer)
    OutgoingFlowSelector.choose_outgoing_flow = staticmethod(selector.choose)
    
    episodes = rl_conf.get("episodes", 10)
    cases = rl_conf.get("total_cases", 50)
    
    for ep in range(episodes):
        safety.resource_budget = rl_conf.get("resource_budget", 1000)
        
        # Simulación: Si hay estado parcial, úsalo
        try:
            if state_path and os.path.exists(state_path):
                # Intentar pasar log_file (soporte varía por versión de Prosimos)
                try:
                    run_simulation(bpmn, json_path, total_cases=cases, log_file=state_path)
                except TypeError:
                    run_simulation(bpmn, json_path, total_cases=cases)
            else:
                run_simulation(bpmn, json_path, total_cases=cases)
        except Exception as e:
            logger.error(f"Error en simulación: {e}")
            
        if (ep+1) % 5 == 0:
            logger.info(f"Episodio {ep+1}/{episodes} - Buffer: {len(buffer.data)}")

    # Guardar
    # Detectar log_name desde el directorio donde estamos guardando (usar el mismo que en main)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    
    # Verificar si existe results/bpi2017_train/rl (indica que usamos train set)
    train_rl_dir = os.path.join(project_root, "results", "bpi2017_train", "rl")
    if os.path.exists(train_rl_dir) or os.path.exists(os.path.join(project_root, "results", "bpi2017_train", "simod")):
        log_name = "bpi2017_train"
    else:
        log_name = get_log_name_from_path(config["log_config"]["log_path"])
    
    out_dir = build_output_path(config["script_config"]["rl_output_dir"], log_name, "rl", "data")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(project_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    out_csv = os.path.join(out_dir, "experience_buffer.csv")
    if buffer.data:
        pd.DataFrame(buffer.data).to_csv(out_csv, index=False)
        logger.info(f"✅ Buffer guardado: {out_csv}")
    else:
        logger.warning("❌ Buffer vacío.")
        
    OutgoingFlowSelector.choose_outgoing_flow = orig # Restaurar

def main():
    config = load_config()
    if not config: return
    
    # Encontrar directorio raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)  # src/
    project_root = os.path.dirname(src_dir)  # proyecto raíz
    
    # Detectar si estamos usando train set: verificar si existe results/bpi2017_train/simod
    train_simod_dir = os.path.join(project_root, "results", "bpi2017_train", "simod")
    train_bpmn = os.path.join(train_simod_dir, "bpi2017_train.bpmn")
    
    if os.path.exists(train_bpmn):
        # Usar train set
        log_name = "bpi2017_train"
        logger.info(f"🎯 Detectado train set: usando {log_name}")
    else:
        # Usar nombre del log desde config.yaml
        log_name = get_log_name_from_path(config["log_config"]["log_path"])
        logger.info(f"📋 Usando nombre del log desde config: {log_name}")
    
    # 1. Buscar Modelo BPMN/JSON (Fase 1)
    simod_dir = build_output_path(config["script_config"]["output_dir"], log_name, "simod", "data")
    if not os.path.isabs(simod_dir):
        simod_dir = os.path.join(project_root, simod_dir)
    bpmn = os.path.join(simod_dir, f"{log_name}.bpmn")
    json_p = os.path.join(simod_dir, f"{log_name}.json")
    
    if not os.path.exists(bpmn):
        logger.error(f"No se encontró BPMN en {simod_dir}. Ejecuta extract_bpmn_json.py")
        return

    # 2. Buscar Estado Parcial (Fase 2A)
    state_dir = build_output_path(config["script_config"]["state_output_dir"], log_name, "state", "data")
    if not os.path.isabs(state_dir):
        state_dir = os.path.join(project_root, state_dir)
    state_file = None
    if os.path.exists(state_dir):
        files = [os.path.join(state_dir, f) for f in os.listdir(state_dir) if "process_state" in f and f.endswith(".json")]
        if files:
            state_file = max(files, key=os.path.getmtime)
            logger.info(f"📍 Usando estado parcial: {os.path.basename(state_file)}")
        else:
            logger.warning("⚠️ No hay estados parciales. Entrenando desde cero.")
            
    train(bpmn, json_p, state_file, config)

if __name__ == "__main__":
    main()