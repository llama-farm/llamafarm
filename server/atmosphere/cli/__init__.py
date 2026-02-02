"""
Atmosphere CLI.

Commands:
    init        Initialize node (create mesh or join with token)
    status      Show mesh status
    nodes       List connected nodes
    token       Manage tokens (create, revoke)
    intent      Route an intent through the mesh
    demo        Interactive demo/test mode
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from .commands import (
    cmd_init,
    cmd_status,
    cmd_nodes,
    cmd_token,
    cmd_intent,
    cmd_start,
)
from .demo import cmd_demo


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="atmosphere",
        description="Atmosphere - The Internet of Intent"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: ~/.atmosphere)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # init
    init_parser = subparsers.add_parser("init", help="Initialize this node")
    init_parser.add_argument(
        "--create-mesh",
        type=str,
        metavar="NAME",
        help="Create a new mesh with this name"
    )
    init_parser.add_argument(
        "--token",
        type=str,
        help="Join existing mesh with this token"
    )
    init_parser.add_argument(
        "--seed",
        type=str,
        action="append",
        help="Seed peer address (host:port)"
    )
    
    # status
    status_parser = subparsers.add_parser("status", help="Show mesh status")
    status_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # nodes
    nodes_parser = subparsers.add_parser("nodes", help="List connected nodes")
    nodes_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # token
    token_parser = subparsers.add_parser("token", help="Manage tokens")
    token_sub = token_parser.add_subparsers(dest="token_action")
    
    token_create = token_sub.add_parser("create", help="Create invite token")
    token_create.add_argument("--name", type=str, help="Token recipient name")
    token_create.add_argument("--capabilities", type=str, nargs="+", help="Granted capabilities")
    token_create.add_argument("--expires", type=int, default=24, help="Expiry hours (default: 24)")
    
    token_revoke = token_sub.add_parser("revoke", help="Revoke a token")
    token_revoke.add_argument("device_id", type=str, help="Device ID to revoke")
    token_revoke.add_argument("--reason", type=str, default="manual", help="Revocation reason")
    
    token_list = token_sub.add_parser("list", help="List issued tokens")
    
    # intent
    intent_parser = subparsers.add_parser("intent", help="Route an intent")
    intent_parser.add_argument("text", type=str, help="Intent text")
    intent_parser.add_argument("--file", type=str, help="Attach file")
    intent_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # start (daemon mode)
    start_parser = subparsers.add_parser("start", help="Start Atmosphere daemon")
    start_parser.add_argument("--port", type=int, default=11450, help="Gossip port")
    start_parser.add_argument("--api-port", type=int, default=11451, help="API port")
    start_parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    
    # demo (interactive test mode)
    demo_parser = subparsers.add_parser("demo", help="Interactive demo/test mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up data dir
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".atmosphere"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dispatch command
    try:
        if args.command == "init":
            asyncio.run(cmd_init(args, data_dir))
        elif args.command == "status":
            asyncio.run(cmd_status(args, data_dir))
        elif args.command == "nodes":
            asyncio.run(cmd_nodes(args, data_dir))
        elif args.command == "token":
            asyncio.run(cmd_token(args, data_dir))
        elif args.command == "intent":
            asyncio.run(cmd_intent(args, data_dir))
        elif args.command == "start":
            asyncio.run(cmd_start(args, data_dir))
        elif args.command == "demo":
            asyncio.run(cmd_demo(args, data_dir))
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        if args.verbose:
            raise
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
