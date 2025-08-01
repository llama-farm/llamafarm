#!/usr/bin/env python3
"""Super simple test chat app for Llama Brain debugging."""

from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import json
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Configuration
LLAMA_BRAIN_API = "http://localhost:8080"
GENERATED_CONFIGS_DIR = Path("../data/configs")

# Global state for debugging
debug_logs = []
current_session_id = None

def log_debug(message, data=None):
    """Add debug log entry."""
    global debug_logs
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "data": data
    }
    debug_logs.append(log_entry)
    # Keep only last 50 logs
    if len(debug_logs) > 50:
        debug_logs.pop(0)
    print(f"[{timestamp}] {message}")
    if data:
        print(f"    Data: {json.dumps(data, indent=2)[:200]}...")

@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    global current_session_id
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        log_debug("Received user message", {"message": user_message})
        
        # Prepare request to Llama Brain
        payload = {"message": user_message}
        if current_session_id:
            payload["session_id"] = current_session_id
        
        log_debug("Sending to Llama Brain API", payload)
        
        # Call Llama Brain API
        response = requests.post(f"{LLAMA_BRAIN_API}/chat", json=payload, timeout=60)
        
        log_debug("Llama Brain API response", {
            "status_code": response.status_code,
            "headers": dict(response.headers)
        })
        
        if response.status_code == 200:
            result = response.json()
            current_session_id = result.get("session_id")
            
            # Log detailed information about the response
            log_debug("Chat response received", {
                "session_id": current_session_id,
                "message_length": len(result.get("message", "")),
                "actions_count": len(result.get("actions", [])),
                "context_count": len(result.get("context_used", []))
            })
            
            # Log RAG context details if available
            if result.get("context_used"):
                log_debug("RAG Context Found", {
                    "contexts": [
                        {
                            "content_preview": ctx.get("content", "")[:100] + "..." if ctx.get("content") else "No content",
                            "metadata": ctx.get("metadata", {}),
                            "score": ctx.get("relevance_score", "No score")
                        }
                        for ctx in result.get("context_used", [])
                    ]
                })
            else:
                log_debug("No RAG Context", {"note": "Either no documents found or RAG search failed"})
            
            # Log actions if any
            if result.get("actions"):
                log_debug("Agent Actions Detected", {
                    "actions": [
                        {
                            "agent": action.get("agent_type"),
                            "action": action.get("action"),
                            "params": action.get("parameters", {})
                        }
                        for action in result.get("actions", [])
                    ]
                })
            
            return jsonify({
                "success": True,
                "response": result.get("message", ""),
                "session_id": current_session_id,
                "actions": result.get("actions", []),
                "suggestions": result.get("suggestions", []),
                "context_used": result.get("context_used", []),
                "confidence": result.get("confidence", 0.8)
            })
        else:
            error_msg = f"Llama Brain API error: {response.status_code}"
            log_debug("API Error", {"error": error_msg, "response": response.text})
            return jsonify({"error": error_msg, "details": response.text}), 500
            
    except requests.exceptions.Timeout:
        error_msg = "Request timeout - Llama Brain API took too long"
        log_debug("Timeout error", {"error": error_msg})
        return jsonify({"error": error_msg}), 408
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Llama Brain API - is the server running?"
        log_debug("Connection error", {"error": error_msg})
        return jsonify({"error": error_msg}), 503
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        log_debug("Unexpected error", {"error": error_msg})
        return jsonify({"error": error_msg}), 500

@app.route('/api/generate-config', methods=['POST'])
def generate_config():
    """Generate configuration using Llama Brain."""
    try:
        data = request.get_json()
        
        log_debug("Config generation request", data)
        
        # Call Llama Brain config API
        response = requests.post(f"{LLAMA_BRAIN_API}/config", json=data, timeout=120)
        
        log_debug("Config generation response", {
            "status_code": response.status_code
        })
        
        if response.status_code == 200:
            result = response.json()
            
            log_debug("Config generated", {
                "success": result.get("success"),
                "file_path": result.get("file_path")
            })
            
            return jsonify(result)
        else:
            error_msg = f"Config generation failed: {response.status_code}"
            log_debug("Config error", {"error": error_msg, "response": response.text})
            return jsonify({"error": error_msg, "details": response.text}), 500
            
    except Exception as e:
        error_msg = f"Config generation error: {str(e)}"
        log_debug("Config generation error", {"error": error_msg})
        return jsonify({"error": error_msg}), 500

@app.route('/api/status')
def api_status():
    """Get Llama Brain API status."""
    try:
        # Check Llama Brain API
        response = requests.get(f"{LLAMA_BRAIN_API}/status", timeout=10)
        
        if response.status_code == 200:
            status_data = response.json()
            log_debug("Status check successful", status_data)
            return jsonify({
                "llama_brain_status": "connected",
                "details": status_data
            })
        else:
            return jsonify({
                "llama_brain_status": "error",
                "error": f"Status check failed: {response.status_code}"
            }), 500
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            "llama_brain_status": "disconnected",
            "error": "Cannot connect to Llama Brain API"
        }), 503
    except Exception as e:
        return jsonify({
            "llama_brain_status": "error", 
            "error": str(e)
        }), 500

@app.route('/api/debug-logs')
def get_debug_logs():
    """Get debug logs for real-time debugging."""
    return jsonify({"logs": debug_logs})

@app.route('/api/clear-logs', methods=['POST'])
def clear_debug_logs():
    """Clear debug logs."""
    global debug_logs
    debug_logs.clear()
    log_debug("Debug logs cleared")
    return jsonify({"success": True})

@app.route('/api/chat-history')
def get_chat_history():
    """Get chat history from Llama Brain."""
    global current_session_id
    
    if not current_session_id:
        return jsonify({"messages": []})
    
    try:
        response = requests.get(f"{LLAMA_BRAIN_API}/chat/{current_session_id}/history", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to get chat history"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generated-configs')
def list_generated_configs():
    """List generated configuration files."""
    try:
        configs = []
        
        if GENERATED_CONFIGS_DIR.exists():
            for config_type in ["models", "rag", "prompts"]:
                type_dir = GENERATED_CONFIGS_DIR / config_type
                if type_dir.exists():
                    for config_file in type_dir.iterdir():
                        if config_file.is_file():
                            configs.append({
                                "name": config_file.name,
                                "type": config_type,
                                "path": str(config_file),
                                "size": config_file.stat().st_size,
                                "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                            })
        
        log_debug("Listed generated configs", {"count": len(configs)})
        return jsonify({"configs": configs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config-content/<path:config_path>')
def get_config_content(config_path):
    """Get content of a generated config file."""
    try:
        # Security: only allow files in the generated configs directory
        full_path = Path(config_path)
        if not str(full_path).startswith(str(GENERATED_CONFIGS_DIR)):
            return jsonify({"error": "Access denied"}), 403
        
        if full_path.exists() and full_path.is_file():
            with open(full_path, 'r') as f:
                content = f.read()
            
            return jsonify({
                "content": content,
                "filename": full_path.name,
                "size": len(content)
            })
        else:
            return jsonify({"error": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Reset the chat session."""
    global current_session_id
    current_session_id = None
    log_debug("Chat session reset")
    return jsonify({"success": True})

if __name__ == '__main__':
    log_debug("Starting test chat app")
    print("🚀 Starting Test Chat App")
    print(f"   App URL: http://localhost:5001")
    print(f"   Llama Brain API: {LLAMA_BRAIN_API}")
    print(f"   Generated configs: {GENERATED_CONFIGS_DIR}")
    print("\nMake sure Llama Brain server is running on port 8080!")
    
    app.run(debug=True, port=5001, host='0.0.0.0')