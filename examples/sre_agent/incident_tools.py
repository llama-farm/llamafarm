from sdk import tool
import json
import random
import time

# Simulated System State
SYSTEM_STATE = {
    "cpu_usage": 85.0,  # High CPU to trigger incident
    "memory_usage": 45.0,
    "services": {
        "payment-gateway": "degraded",
        "auth-service": "healthy",
        "database-primary": "healthy"
    }
}

@tool(description="Get current system metrics including CPU and Memory usage.")
def get_system_metrics() -> str:
    """
    Returns the current health metrics of the system.
    """
    return json.dumps({
        "timestamp": time.time(),
        "cpu": SYSTEM_STATE["cpu_usage"],
        "memory": SYSTEM_STATE["memory_usage"],
        "disk": 60.0
    })

@tool(description="Fetch recent logs for a specific service.")
def fetch_recent_logs(service_name: str, lines: int = 5) -> str:
    """
    Retrieves the last N lines of logs for a service.
    """
    if service_name not in SYSTEM_STATE["services"]:
        return f"Service '{service_name}' not found."
    
    status = SYSTEM_STATE["services"][service_name]
    lines_to_return = []
    if status == "degraded":
        all_logs = [
            f"[ERROR] {service_name}: Connection timeout",
            f"[ERROR] {service_name}: Retrying transaction",
            f"[WARN] High latency detected",
            f"[ERROR] {service_name}: Connection reset",
            f"[WARN] Garbage collection pause",
            f"[ERROR] {service_name}: 503 Service Unavailable"
        ]
        lines_to_return = all_logs[:lines]
    else:
        all_logs = [
            f"[INFO] {service_name}: Health check passed",
            f"[INFO] {service_name}: Request processed in 20ms",
            f"[INFO] {service_name}: Cache hit",
            f"[INFO] {service_name}: Transaction committed",
            f"[INFO] {service_name}: Worker pool healthy"
        ]
        lines_to_return = all_logs[:lines]
    
    return "\n".join(lines_to_return)

@tool(description="Restart a service to attempt recovery.")
def restart_service(service_name: str) -> str:
    """
    Restarts the specified service. Use this if a service is degraded or failing.
    """
    if service_name not in SYSTEM_STATE["services"]:
        return f"Service '{service_name}' not found."
    
    # Simulate fix
    SYSTEM_STATE["services"][service_name] = "healthy"
    SYSTEM_STATE["cpu_usage"] = 30.0 # CPU cools down
    
    return f"Service '{service_name}' successfully restarted. New status: healthy."

@tool(description="Escalate the incident to the on-call engineer.")
def escalate_incident(reason: str) -> str:
    """
    Escalates the issue if it cannot be resolved automatically.
    """
    return f"INCIDENT ESCALATED: {reason}. Page sent to on-call."
