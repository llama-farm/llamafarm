"""
CLI command implementations.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Lazy imports to avoid slow startup
def _get_service():
    from ..service import AtmosphereService
    return AtmosphereService

def _get_auth():
    from ..auth.rownd_local import RowndLocalService
    return RowndLocalService


async def cmd_init(args, data_dir: Path):
    """Initialize Atmosphere node."""
    config_file = data_dir / "config.json"
    
    if config_file.exists() and not (args.create_mesh or args.token):
        print(f"Already initialized. Config at {config_file}")
        print("Use --create-mesh or --token to reinitialize.")
        return
    
    RowndLocalService = _get_auth()
    auth_dir = data_dir / "auth"
    auth_dir.mkdir(exist_ok=True)
    
    if args.create_mesh:
        # Create new mesh
        print(f"Creating new mesh: {args.create_mesh}")
        
        service = RowndLocalService(str(auth_dir))
        service.create_mesh(args.create_mesh)
        
        config = {
            "mesh_name": args.create_mesh,
            "mesh_id": service.mesh.mesh_id,
            "node_id": service.device.device_id,
            "role": "founder",
            "seeds": args.seed or [],
            "port": 11450,
        }
        
        print(f"✓ Mesh created: {config['mesh_id'][:12]}...")
        print(f"✓ Node ID: {config['node_id'][:12]}...")
        
        # Generate initial invite token
        token = service.create_invite_token(
            recipient_name="initial-invite",
            capabilities=["*"],
            validity_hours=720  # 30 days
        )
        
        print(f"\n🔑 Invite token (share this to add nodes):\n")
        print(f"  {token}\n")
        
    elif args.token:
        # Join existing mesh
        print("Joining existing mesh...")
        
        service = RowndLocalService(str(auth_dir))
        
        try:
            result = service.join_with_token(args.token, args.seed or [])
            
            config = {
                "mesh_name": result.get("mesh_name", "unknown"),
                "mesh_id": result.get("mesh_id"),
                "node_id": service.device.device_id,
                "role": "member",
                "seeds": args.seed or [],
                "port": 11450,
            }
            
            print(f"✓ Joined mesh: {config['mesh_name']}")
            print(f"✓ Node ID: {config['node_id'][:12]}...")
            
        except Exception as e:
            print(f"Failed to join mesh: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("Specify --create-mesh NAME or --token TOKEN", file=sys.stderr)
        sys.exit(1)
    
    # Save config
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Config saved to {config_file}")
    print("\nNext steps:")
    print("  atmosphere start     # Start the daemon")
    print("  atmosphere status    # Check mesh status")


async def cmd_status(args, data_dir: Path):
    """Show mesh status."""
    config_file = data_dir / "config.json"
    
    if not config_file.exists():
        print("Not initialized. Run: atmosphere init --create-mesh NAME")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Try to connect to running daemon
    try:
        import aiohttp
        api_port = config.get("api_port", 11451)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{api_port}/status") as resp:
                if resp.status == 200:
                    status = await resp.json()
                    
                    if args.json:
                        print(json.dumps(status, indent=2))
                    else:
                        print(f"Mesh:     {status['mesh_name']} ({status['mesh_id'][:12]}...)")
                        print(f"Node:     {status['node_id'][:12]}...")
                        print(f"Role:     {status['role']}")
                        print(f"Peers:    {status['peer_count']}")
                        print(f"Status:   {'🟢 Online' if status['connected'] else '🔴 Offline'}")
                        
                        if status.get('capabilities'):
                            print(f"\nCapabilities:")
                            for cap in status['capabilities']:
                                print(f"  - {cap}")
                    return
    except Exception:
        pass
    
    # Daemon not running, show config
    if args.json:
        print(json.dumps({"config": config, "daemon": "not_running"}, indent=2))
    else:
        print(f"Mesh:     {config.get('mesh_name', 'unknown')}")
        print(f"Node:     {config.get('node_id', 'unknown')[:12]}...")
        print(f"Role:     {config.get('role', 'unknown')}")
        print(f"Status:   🔴 Daemon not running")
        print("\nRun: atmosphere start")


async def cmd_nodes(args, data_dir: Path):
    """List connected nodes."""
    config_file = data_dir / "config.json"
    
    if not config_file.exists():
        print("Not initialized.", file=sys.stderr)
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    try:
        import aiohttp
        api_port = config.get("api_port", 11451)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{api_port}/nodes") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if args.json:
                        print(json.dumps(data, indent=2))
                    else:
                        nodes = data.get("nodes", [])
                        if not nodes:
                            print("No other nodes connected.")
                        else:
                            print(f"{'Node ID':<20} {'Address':<25} {'Status':<10} {'Capabilities'}")
                            print("-" * 80)
                            for node in nodes:
                                status = "🟢" if node.get("connected") else "🔴"
                                caps = ", ".join(node.get("capabilities", [])[:3])
                                if len(node.get("capabilities", [])) > 3:
                                    caps += "..."
                                print(f"{node['id'][:18]:<20} {node.get('address', 'unknown'):<25} {status:<10} {caps}")
                    return
    except Exception as e:
        print(f"Error connecting to daemon: {e}", file=sys.stderr)
        print("Is the daemon running? Try: atmosphere start")
        sys.exit(1)


async def cmd_token(args, data_dir: Path):
    """Manage tokens."""
    config_file = data_dir / "config.json"
    
    if not config_file.exists():
        print("Not initialized.", file=sys.stderr)
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    RowndLocalService = _get_auth()
    auth_dir = data_dir / "auth"
    service = RowndLocalService(str(auth_dir))
    
    if args.token_action == "create":
        # Create invite token
        name = args.name or "unnamed"
        capabilities = args.capabilities or ["*"]
        expires = args.expires
        
        token = service.create_invite_token(
            recipient_name=name,
            capabilities=capabilities,
            validity_hours=expires
        )
        
        print(f"🔑 Token created for '{name}'")
        print(f"   Capabilities: {', '.join(capabilities)}")
        print(f"   Expires: {expires} hours")
        print(f"\n{token}\n")
        
    elif args.token_action == "revoke":
        # Revoke a device
        device_id = args.device_id
        reason = args.reason
        
        service.revoke_device(device_id, reason)
        print(f"✓ Revoked device: {device_id}")
        print(f"  Reason: {reason}")
        
    elif args.token_action == "list":
        # List issued tokens
        tokens = service.list_issued_tokens()
        if not tokens:
            print("No tokens issued.")
        else:
            print(f"{'Recipient':<20} {'Device ID':<20} {'Issued':<20} {'Status'}")
            print("-" * 80)
            for t in tokens:
                print(f"{t['recipient'][:18]:<20} {t['device_id'][:18]:<20} {t['issued']:<20} {t['status']}")
    
    else:
        print("Specify action: create, revoke, or list", file=sys.stderr)
        sys.exit(1)


async def cmd_intent(args, data_dir: Path):
    """Route an intent through the mesh."""
    config_file = data_dir / "config.json"
    
    if not config_file.exists():
        print("Not initialized.", file=sys.stderr)
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    try:
        import aiohttp
        api_port = config.get("api_port", 11451)
        
        payload = {"intent": args.text}
        
        # Handle file attachment
        if args.file:
            with open(args.file, "rb") as f:
                import base64
                payload["attachment"] = {
                    "filename": args.file,
                    "data": base64.b64encode(f.read()).decode()
                }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{api_port}/intent",
                json=payload
            ) as resp:
                data = await resp.json()
                
                if args.json:
                    print(json.dumps(data, indent=2))
                else:
                    if data.get("error"):
                        print(f"Error: {data['error']}", file=sys.stderr)
                        sys.exit(1)
                    
                    print(f"Routed to: {data.get('routed_to', 'local')}")
                    print(f"Latency:   {data.get('latency_ms', 0):.1f}ms")
                    print()
                    print(data.get("result", "No result"))
                    
    except aiohttp.ClientConnectorError:
        print("Daemon not running. Start with: atmosphere start", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_start(args, data_dir: Path):
    """Start Atmosphere daemon."""
    config_file = data_dir / "config.json"
    
    if not config_file.exists():
        print("Not initialized. Run: atmosphere init --create-mesh NAME")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    AtmosphereService = _get_service()
    
    service = AtmosphereService(
        data_dir=data_dir,
        gossip_port=args.port,
        api_port=args.api_port
    )
    
    print(f"Starting Atmosphere daemon...")
    print(f"  Mesh:        {config.get('mesh_name')}")
    print(f"  Gossip port: {args.port}")
    print(f"  API port:    {args.api_port}")
    
    if args.foreground:
        print("\nRunning in foreground (Ctrl+C to stop)...")
        await service.run_forever()
    else:
        # TODO: Proper daemonization
        print("\nForeground mode required for now. Use -f flag.")
        print("  atmosphere start -f")
        sys.exit(1)
