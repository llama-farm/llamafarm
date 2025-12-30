"""Needle Real: Universal Agent Framework with REAL AI.

This script demonstrates a fully functional Agent system that:
1. Receives external data via a tool (Simulated Sensor).
2. Trains a real Anomaly Detection model on the Universal Runtime.
3. Continuously monitors incoming data and triggers alerts using live inference.
"""

import asyncio
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path

# Add Custom Runtime path to find 'sdk' module
# ../../../runtimes/custom relative to examples/agent_framework_demo/src/needle_real.py
runtime_path = Path(__file__).parents[3] / "runtimes" / "custom"
if str(runtime_path) not in sys.path:
    sys.path.append(str(runtime_path))

try:
    from sdk import tool, Agent, LlamaFarmClient
except ImportError:
    # Fallback for when running in different contexts
    from llamafarm.sdk import tool, Agent, LlamaFarmClient

# --- State Management ---

@dataclass
class SystemState:
    """Encapsulates the shared state of the squad monitoring system."""
    data_buffer: List[Dict[str, float]] = field(default_factory=list)
    baseline_data: List[List[float]] = field(default_factory=list)
    model_name: str = "bio_v1"
    is_trained: bool = False
    
    def add_reading(self, reading: Dict[str, float]) -> str:
        if not self.is_trained:
            # Accumulate training data
            vector = [reading["heart_rate"], reading["spo2"], reading["body_temp"]]
            self.baseline_data.append(vector)
            return f"Data recorded for training. Total samples: {len(self.baseline_data)}"
        else:
            # Buffer for live inference
            self.data_buffer.append(reading)
            return "Vitals received and buffered for analysis."

STATE = SystemState()

# --- Tools ---

@tool
def inject_vitals(heart_rate: int, spo2: int, body_temp: float) -> str:
    """Simulate receiving data from a soldier's biosensor.
    
    This tool allows external systems or agents to pipe data into the framework.
    """
    reading = {"heart_rate": float(heart_rate), "spo2": float(spo2), "body_temp": float(body_temp)}
    return STATE.add_reading(reading)

@tool
def request_medevac(reason: str) -> str:
    """Trigger a Medevac request for critical situations."""
    print(f"!!! CRITICAL ACTION !!! MEDEVAC DISPATCHED: {reason}")
    return "Medevac en route."

# --- Active Agents ---

class BioMonitor(Agent):
    """Monitors incoming vitals using Universal Runtime AI."""
    
    interval = 1.0
    
    async def on_tick(self):
        # 1. Check if model needs training
        if not STATE.is_trained and len(STATE.baseline_data) >= 50:
            print(f"[{self.name}] Sufficient baseline data ({len(STATE.baseline_data)}). Training model...")
            try:
                # Call Universal Runtime to train
                await self.client.post("/v1/anomaly/fit", json={
                    "model": STATE.model_name,
                    "data": STATE.baseline_data,
                    "model_type": "isolation_forest"
                })
                STATE.is_trained = True
                print(f"[{self.name}] Model '{STATE.model_name}' trained successfully!")
            except Exception as e:
                print(f"[{self.name}] Training failed: {e}")
                return

        # 2. Process Buffered Data
        if STATE.is_trained and STATE.data_buffer:
            # Process one reading at a time (FIFO)
            data = STATE.data_buffer.pop(0)
            vector = [data["heart_rate"], data["spo2"], data["body_temp"]]
            
            try:
                # Call Universal Runtime for Inference
                response = await self.client.post("/v1/anomaly/detect", json={
                    "model": STATE.model_name,
                    "data": [vector]
                })
                
                # Parse response
                # Format: {"predictions": [-1], "scores": [0.85]} where -1 is anomaly
                is_anomaly = response.get("predictions", [1])[0] == -1
                score = response.get("scores", [0.0])[0]
                
                print(f"[{self.name}] Analyzing {data} -> Anomaly: {is_anomaly} (Score: {score:.3f})")
                
                if is_anomaly:
                    print(f"[{self.name}] ANOMALY DETECTED!")
                    # Use heuristic backup for severity
                    if data["heart_rate"] > 120 or data["spo2"] < 90:
                        request_medevac(f"AI Detected Critical Vitals: {data}")
                        
            except Exception as e:
                print(f"[{self.name}] Inference failed: {e}")

class DataPipe(Agent):
    """Simulates an external data stream piping data into the system."""
    
    interval = 0.1 # Fast simulation
    _tick_count = 0
    
    async def on_tick(self):
        self._tick_count += 1
        
        # Phase 1: Training Data (Normal) - Generate 60 samples
        if self._tick_count <= 60:
            hr = random.randint(60, 80)
            spo2 = random.randint(97, 99)
            temp = random.uniform(36.5, 37.5)
            
            if self._tick_count % 10 == 0:
                print(f"[{self.name}] Piping Training Data: HR={hr} (Sample {self._tick_count})")
            inject_vitals(hr, spo2, temp)
            
        # Phase 2: Wait for Training
        elif self._tick_count <= 70:
             if self._tick_count % 5 == 0:
                 print(f"[{self.name}] Pausing for training...")
             
        # Phase 3: Anomaly Injection
        elif self._tick_count == 72:
            print(f"[{self.name}] !!! INJECTING ANOMALY !!!")
            # Severe anomaly
            inject_vitals(160, 80, 40.0)
            
        # Phase 4: Stop
        elif self._tick_count > 75:
            print(f"[{self.name}] Simulation complete.")
            self._is_running = False # Stop self
