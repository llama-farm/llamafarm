"""
Interactive demo mode for testing Atmosphere mesh.

This creates a simulated multi-node mesh environment for testing
routing, gossip, and failure scenarios.
"""

import asyncio
import json
import sys
import time
import uuid
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Lazy imports to avoid slow startup
def _get_mesh_classes():
    from ..mesh.node import Node, NodeIdentity, MeshIdentity
    from ..mesh.gossip import GossipProtocol, CapabilityInfo, ResourceInfo
    from ..router.gradient import GradientTable, GradientEntry
    return Node, NodeIdentity, MeshIdentity, GossipProtocol, CapabilityInfo, ResourceInfo, GradientTable, GradientEntry

def _get_auth():
    from ..auth.identity import KeyPair
    return KeyPair


console = Console()


class SimulatedNode:
    """A simulated node for demo purposes."""
    
    def __init__(self, node_id: str, name: str, capabilities: List[dict]):
        self.node_id = node_id
        self.name = name
        self.capabilities = capabilities
        self.alive = True
        self.peers: List[str] = []
        self.gradient_table: Dict[str, dict] = {}
        self.intent_count = 0
        self.last_gossip = time.time()
        
        # Build gradient table from capabilities
        for cap in capabilities:
            self.gradient_table[cap["id"]] = {
                "capability_id": cap["id"],
                "label": cap["label"],
                "hops": 0,
                "via": "local",
                "latency_ms": 5,
                "vector": cap["vector"]
            }
    
    def handle_intent(self, intent: dict) -> dict:
        """Handle an intent."""
        self.intent_count += 1
        return {
            "handled_by": self.node_id,
            "handler_name": self.name,
            "result": f"Intent processed by {self.name}",
            "latency_ms": np.random.uniform(10, 50)
        }
    
    def gossip_tick(self):
        """Simulate gossip propagation."""
        self.last_gossip = time.time()


class DemoMesh:
    """A simulated mesh for demo purposes."""
    
    def __init__(self):
        self.nodes: Dict[str, SimulatedNode] = {}
        self.mesh_id = str(uuid.uuid4())[:16]
        self.mesh_name = "demo-mesh"
        self.invite_tokens: List[dict] = []
        self.intent_log: List[dict] = []
    
    def create_test_mesh(self):
        """Create a mesh with 3 test nodes."""
        # Node 1: Code generation specialist
        node1 = SimulatedNode(
            node_id=str(uuid.uuid4())[:16],
            name="CodeBot",
            capabilities=[
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "code_generation",
                    "vector": [0.9, 0.1, 0.2, 0.8, 0.3]
                },
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "python_expert",
                    "vector": [0.8, 0.2, 0.1, 0.9, 0.2]
                }
            ]
        )
        
        # Node 2: Vision/image specialist
        node2 = SimulatedNode(
            node_id=str(uuid.uuid4())[:16],
            name="VisionBot",
            capabilities=[
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "image_analysis",
                    "vector": [0.1, 0.9, 0.8, 0.2, 0.7]
                },
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "object_detection",
                    "vector": [0.2, 0.85, 0.75, 0.1, 0.8]
                }
            ]
        )
        
        # Node 3: General chat/conversation
        node3 = SimulatedNode(
            node_id=str(uuid.uuid4())[:16],
            name="ChatBot",
            capabilities=[
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "conversation",
                    "vector": [0.5, 0.5, 0.9, 0.6, 0.4]
                },
                {
                    "id": str(uuid.uuid4())[:12],
                    "label": "general_qa",
                    "vector": [0.6, 0.4, 0.85, 0.5, 0.5]
                }
            ]
        )
        
        # Connect them in a mesh
        node1.peers = [node2.node_id, node3.node_id]
        node2.peers = [node1.node_id, node3.node_id]
        node3.peers = [node1.node_id, node2.node_id]
        
        self.nodes[node1.node_id] = node1
        self.nodes[node2.node_id] = node2
        self.nodes[node3.node_id] = node3
        
        # Propagate gradient info
        self._propagate_gradients()
        
        return [node1, node2, node3]
    
    def _propagate_gradients(self):
        """Simulate gradient table propagation across mesh."""
        for node in self.nodes.values():
            for peer_id in node.peers:
                peer = self.nodes.get(peer_id)
                if peer and peer.alive:
                    # Share capabilities with peer
                    for cap_id, gradient in peer.gradient_table.items():
                        if cap_id not in node.gradient_table or node.gradient_table[cap_id]["hops"] > gradient["hops"] + 1:
                            node.gradient_table[cap_id] = {
                                **gradient,
                                "hops": gradient["hops"] + 1,
                                "via": peer_id,
                                "latency_ms": gradient["latency_ms"] + 10
                            }
    
    def add_node(self, name: str, capabilities: List[dict]) -> SimulatedNode:
        """Add a new node to the mesh."""
        node = SimulatedNode(
            node_id=str(uuid.uuid4())[:16],
            name=name,
            capabilities=capabilities
        )
        
        # Connect to existing nodes
        if self.nodes:
            existing_ids = list(self.nodes.keys())
            node.peers = existing_ids[:2]  # Connect to first 2
            for peer_id in node.peers:
                self.nodes[peer_id].peers.append(node.node_id)
        
        self.nodes[node.node_id] = node
        self._propagate_gradients()
        
        return node
    
    def route_intent(self, intent_text: str) -> dict:
        """Route an intent to the best matching node."""
        # Simple semantic routing based on keywords
        intent_vector = self._vectorize_intent(intent_text)
        
        best_node = None
        best_score = -1
        best_capability = None
        
        # Find best match across all nodes
        for node in self.nodes.values():
            if not node.alive:
                continue
            
            for cap_id, gradient in node.gradient_table.items():
                cap_vector = np.array(gradient["vector"])
                score = np.dot(intent_vector, cap_vector) / (np.linalg.norm(intent_vector) * np.linalg.norm(cap_vector))
                
                # Penalize by hops and latency
                adjusted_score = score - (gradient["hops"] * 0.05) - (gradient["latency_ms"] * 0.0001)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_node = node
                    best_capability = gradient
        
        if not best_node:
            return {"error": "No capable node found"}
        
        # Route the intent
        result = best_node.handle_intent({"text": intent_text, "vector": intent_vector.tolist()})
        
        log_entry = {
            "timestamp": time.time(),
            "intent": intent_text,
            "routed_to": best_node.node_id,
            "node_name": best_node.name,
            "capability": best_capability["label"],
            "hops": best_capability["hops"],
            "score": float(best_score),
            "latency_ms": result["latency_ms"]
        }
        
        self.intent_log.append(log_entry)
        
        return {
            "success": True,
            "routed_to": best_node.node_id,
            "node_name": best_node.name,
            "capability": best_capability["label"],
            "hops": best_capability["hops"],
            "score": best_score,
            "result": result["result"],
            "latency_ms": result["latency_ms"]
        }
    
    def _vectorize_intent(self, text: str) -> np.ndarray:
        """Simple keyword-based vectorization for demo."""
        text_lower = text.lower()
        
        # [code_weight, vision_weight, chat_weight, technical_weight, creative_weight]
        vector = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        
        # Code keywords
        if any(word in text_lower for word in ["code", "function", "python", "javascript", "program", "debug", "algorithm"]):
            vector[0] += 0.5
            vector[3] += 0.3
        
        # Vision keywords
        if any(word in text_lower for word in ["image", "photo", "picture", "detect", "vision", "see", "visual", "recognize"]):
            vector[1] += 0.5
            vector[3] += 0.2
        
        # Chat keywords
        if any(word in text_lower for word in ["chat", "talk", "conversation", "discuss", "tell me", "what do you think"]):
            vector[2] += 0.5
            vector[4] += 0.2
        
        # Technical
        if any(word in text_lower for word in ["api", "database", "server", "optimize", "architecture", "system"]):
            vector[3] += 0.4
        
        # Creative
        if any(word in text_lower for word in ["create", "design", "story", "imagine", "creative", "art"]):
            vector[4] += 0.4
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def generate_token(self, name: str = "demo-invite") -> str:
        """Generate an invite token."""
        token = {
            "token": f"atmos_{str(uuid.uuid4())[:24]}",
            "mesh_id": self.mesh_id,
            "mesh_name": self.mesh_name,
            "recipient": name,
            "created": time.time(),
            "expires": time.time() + 86400  # 24 hours
        }
        self.invite_tokens.append(token)
        return token["token"]


async def cmd_demo(args, data_dir: Path):
    """Run interactive demo mode."""
    mesh = DemoMesh()
    
    console.print(Panel.fit(
        "[bold cyan]🌐 Atmosphere Mesh - Interactive Demo[/bold cyan]\n"
        "[dim]Test routing, gossip, and failure scenarios[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    while True:
        # Display menu
        console.print("[bold yellow]═══ Demo Menu ═══[/bold yellow]")
        console.print("[cyan]a)[/cyan] Create test mesh (3 nodes)")
        console.print("[cyan]b)[/cyan] Add node")
        console.print("[cyan]c)[/cyan] List nodes")
        console.print("[cyan]d)[/cyan] Send intent")
        console.print("[cyan]e)[/cyan] Kill node")
        console.print("[cyan]f)[/cyan] Revive node")
        console.print("[cyan]g)[/cyan] Show routes (gradient table)")
        console.print("[cyan]h)[/cyan] Generate token")
        console.print("[cyan]i)[/cyan] Join mesh (simulate)")
        console.print("[cyan]j)[/cyan] Show intent log")
        console.print("[cyan]q)[/cyan] Quit")
        console.print()
        
        choice = Prompt.ask("[bold]Choose option[/bold]", default="a").lower()
        console.print()
        
        if choice == "q":
            console.print("[yellow]👋 Goodbye![/yellow]")
            break
        
        elif choice == "a":
            # Create test mesh
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Creating test mesh...", total=None)
                nodes = mesh.create_test_mesh()
                await asyncio.sleep(0.5)  # Dramatic pause
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Created mesh '{mesh.mesh_name}' with 3 nodes![/green]")
            console.print()
            
            # Show nodes
            table = Table(title="🤖 Mesh Nodes", box=box.ROUNDED)
            table.add_column("Node", style="cyan")
            table.add_column("ID", style="dim")
            table.add_column("Capabilities", style="yellow")
            table.add_column("Status", justify="center")
            
            for node in nodes:
                caps = ", ".join(c["label"] for c in node.capabilities)
                table.add_row(
                    node.name,
                    node.node_id[:12] + "...",
                    caps,
                    "[green]🟢 Online[/green]"
                )
            
            console.print(table)
            console.print()
        
        elif choice == "b":
            # Add node
            if not mesh.nodes:
                console.print("[red]⚠ Create a mesh first (option a)[/red]")
                continue
            
            name = Prompt.ask("[cyan]Node name[/cyan]", default=f"Node-{len(mesh.nodes) + 1}")
            
            console.print("[cyan]Capability (or 'done' to finish):[/cyan]")
            console.print("  [dim]Examples: audio_transcription, video_processing, data_analysis[/dim]")
            
            capabilities = []
            while True:
                cap_name = Prompt.ask("  Capability label", default="done")
                if cap_name.lower() == "done":
                    break
                
                # Generate semi-random vector
                vector = np.random.rand(5).tolist()
                capabilities.append({
                    "id": str(uuid.uuid4())[:12],
                    "label": cap_name,
                    "vector": vector
                })
            
            if not capabilities:
                console.print("[yellow]No capabilities added, using default[/yellow]")
                capabilities = [{
                    "id": str(uuid.uuid4())[:12],
                    "label": "general_purpose",
                    "vector": [0.5, 0.5, 0.5, 0.5, 0.5]
                }]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Adding node to mesh...", total=None)
                await asyncio.sleep(0.3)
                node = mesh.add_node(name, capabilities)
                await asyncio.sleep(0.2)
                task = progress.add_task("[cyan]Propagating gradients...", total=None)
                await asyncio.sleep(0.3)
            
            console.print(f"[green]✓ Node '{name}' added successfully![/green]")
            console.print(f"[dim]  Node ID: {node.node_id}[/dim]")
            console.print(f"[dim]  Peers: {len(node.peers)}[/dim]")
            console.print()
        
        elif choice == "c":
            # List nodes
            if not mesh.nodes:
                console.print("[yellow]No nodes in mesh yet[/yellow]")
                continue
            
            table = Table(title="🤖 Mesh Nodes", box=box.ROUNDED)
            table.add_column("Node", style="cyan")
            table.add_column("ID", style="dim")
            table.add_column("Capabilities", style="yellow")
            table.add_column("Peers", justify="center")
            table.add_column("Intents", justify="center")
            table.add_column("Status", justify="center")
            
            for node in mesh.nodes.values():
                caps = ", ".join(c["label"] for c in node.capabilities)
                status = "[green]🟢 Online[/green]" if node.alive else "[red]🔴 Offline[/red]"
                table.add_row(
                    node.name,
                    node.node_id[:12] + "...",
                    caps,
                    str(len(node.peers)),
                    str(node.intent_count),
                    status
                )
            
            console.print(table)
            console.print()
        
        elif choice == "d":
            # Send intent
            if not mesh.nodes:
                console.print("[red]⚠ Create a mesh first (option a)[/red]")
                continue
            
            console.print("[cyan]Example intents:[/cyan]")
            console.print("  [dim]• 'Write a Python function to sort a list'[/dim]")
            console.print("  [dim]• 'Analyze this image for objects'[/dim]")
            console.print("  [dim]• 'Let's have a conversation about AI'[/dim]")
            console.print()
            
            intent = Prompt.ask("[bold cyan]Enter intent[/bold cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]🧭 Routing intent...", total=None)
                await asyncio.sleep(0.4)
                result = mesh.route_intent(intent)
                await asyncio.sleep(0.2)
            
            if "error" in result:
                console.print(f"[red]✗ {result['error']}[/red]")
            else:
                panel_content = f"""[green]✓ Intent routed successfully![/green]

[cyan]Destination:[/cyan] {result['node_name']} ({result['routed_to'][:12]}...)
[cyan]Capability Match:[/cyan] {result['capability']}
[cyan]Confidence Score:[/cyan] {result['score']:.3f}
[cyan]Hops:[/cyan] {result['hops']}
[cyan]Latency:[/cyan] {result['latency_ms']:.1f}ms

[yellow]Result:[/yellow]
{result['result']}"""
                
                console.print(Panel(panel_content, title="🎯 Routing Result", border_style="green"))
            console.print()
        
        elif choice == "e":
            # Kill node
            if not mesh.nodes:
                console.print("[red]⚠ Create a mesh first (option a)[/red]")
                continue
            
            alive_nodes = [n for n in mesh.nodes.values() if n.alive]
            if not alive_nodes:
                console.print("[yellow]No alive nodes to kill[/yellow]")
                continue
            
            console.print("[cyan]Alive nodes:[/cyan]")
            for i, node in enumerate(alive_nodes, 1):
                console.print(f"  {i}) {node.name} ({node.node_id[:12]}...)")
            
            choice_num = Prompt.ask("Which node to kill?", default="1")
            try:
                idx = int(choice_num) - 1
                node = alive_nodes[idx]
                node.alive = False
                
                console.print(f"[red]💀 Node '{node.name}' is now offline[/red]")
                console.print(f"[dim]  Gradient tables will exclude this node from routing[/dim]")
            except (ValueError, IndexError):
                console.print("[red]Invalid choice[/red]")
            console.print()
        
        elif choice == "f":
            # Revive node
            if not mesh.nodes:
                console.print("[red]⚠ Create a mesh first (option a)[/red]")
                continue
            
            dead_nodes = [n for n in mesh.nodes.values() if not n.alive]
            if not dead_nodes:
                console.print("[yellow]No dead nodes to revive[/yellow]")
                continue
            
            console.print("[cyan]Dead nodes:[/cyan]")
            for i, node in enumerate(dead_nodes, 1):
                console.print(f"  {i}) {node.name} ({node.node_id[:12]}...)")
            
            choice_num = Prompt.ask("Which node to revive?", default="1")
            try:
                idx = int(choice_num) - 1
                node = dead_nodes[idx]
                node.alive = True
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"[cyan]Reviving {node.name}...", total=None)
                    await asyncio.sleep(0.3)
                    task = progress.add_task("[cyan]Re-syncing gradients...", total=None)
                    await asyncio.sleep(0.3)
                
                console.print(f"[green]✓ Node '{node.name}' is back online![/green]")
            except (ValueError, IndexError):
                console.print("[red]Invalid choice[/red]")
            console.print()
        
        elif choice == "g":
            # Show gradient table
            if not mesh.nodes:
                console.print("[red]⚠ Create a mesh first (option a)[/red]")
                continue
            
            # Pick a node to show
            node_list = list(mesh.nodes.values())
            console.print("[cyan]Select node to view gradient table:[/cyan]")
            for i, node in enumerate(node_list, 1):
                console.print(f"  {i}) {node.name}")
            
            choice_num = Prompt.ask("Node number", default="1")
            try:
                idx = int(choice_num) - 1
                node = node_list[idx]
                
                table = Table(
                    title=f"🗺️ Gradient Table for {node.name}",
                    box=box.ROUNDED
                )
                table.add_column("Capability", style="yellow")
                table.add_column("Hops", justify="center")
                table.add_column("Via", style="cyan")
                table.add_column("Latency", justify="right")
                table.add_column("Vector (first 3)", style="dim")
                
                for cap_id, gradient in node.gradient_table.items():
                    via = "local" if gradient["hops"] == 0 else gradient["via"][:12] + "..."
                    vector_str = ", ".join(f"{v:.2f}" for v in gradient["vector"][:3])
                    
                    hop_style = "green" if gradient["hops"] == 0 else "yellow" if gradient["hops"] <= 1 else "red"
                    
                    table.add_row(
                        gradient["label"],
                        f"[{hop_style}]{gradient['hops']}[/{hop_style}]",
                        via,
                        f"{gradient['latency_ms']:.0f}ms",
                        vector_str
                    )
                
                console.print(table)
            except (ValueError, IndexError):
                console.print("[red]Invalid choice[/red]")
            console.print()
        
        elif choice == "h":
            # Generate token
            name = Prompt.ask("[cyan]Token recipient name[/cyan]", default="demo-user")
            token = mesh.generate_token(name)
            
            console.print(Panel.fit(
                f"[green]🔑 Token Generated![/green]\n\n"
                f"[cyan]Recipient:[/cyan] {name}\n"
                f"[cyan]Mesh:[/cyan] {mesh.mesh_name}\n"
                f"[cyan]Expires:[/cyan] 24 hours\n\n"
                f"[bold yellow]Token:[/bold yellow]\n{token}",
                title="Invite Token",
                border_style="green"
            ))
            console.print()
        
        elif choice == "i":
            # Join mesh (simulate)
            if not mesh.invite_tokens:
                console.print("[yellow]No tokens generated yet. Use option 'h' first.[/yellow]")
                continue
            
            token = mesh.invite_tokens[-1]["token"]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Validating token...", total=None)
                await asyncio.sleep(0.3)
                task = progress.add_task("[cyan]Contacting seed peers...", total=None)
                await asyncio.sleep(0.4)
                task = progress.add_task("[cyan]Syncing mesh state...", total=None)
                await asyncio.sleep(0.4)
                task = progress.add_task("[cyan]Propagating capabilities...", total=None)
                await asyncio.sleep(0.3)
            
            new_name = f"NewNode-{len(mesh.nodes) + 1}"
            capabilities = [{
                "id": str(uuid.uuid4())[:12],
                "label": "web_search",
                "vector": np.random.rand(5).tolist()
            }]
            
            node = mesh.add_node(new_name, capabilities)
            
            console.print(Panel(
                f"[green]✓ Successfully joined mesh![/green]\n\n"
                f"[cyan]Node Name:[/cyan] {new_name}\n"
                f"[cyan]Node ID:[/cyan] {node.node_id}\n"
                f"[cyan]Mesh:[/cyan] {mesh.mesh_name}\n"
                f"[cyan]Connected Peers:[/cyan] {len(node.peers)}",
                title="🌐 Mesh Join Complete",
                border_style="green"
            ))
            console.print()
        
        elif choice == "j":
            # Show intent log
            if not mesh.intent_log:
                console.print("[yellow]No intents logged yet[/yellow]")
                continue
            
            table = Table(title="📊 Intent Routing Log", box=box.ROUNDED)
            table.add_column("Time", style="dim")
            table.add_column("Intent", style="cyan")
            table.add_column("Routed To", style="yellow")
            table.add_column("Capability", style="green")
            table.add_column("Hops", justify="center")
            table.add_column("Score", justify="right")
            table.add_column("Latency", justify="right")
            
            for entry in mesh.intent_log[-10:]:  # Last 10
                timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
                intent_short = entry["intent"][:30] + "..." if len(entry["intent"]) > 30 else entry["intent"]
                
                table.add_row(
                    timestamp,
                    intent_short,
                    entry["node_name"],
                    entry["capability"],
                    str(entry["hops"]),
                    f"{entry['score']:.3f}",
                    f"{entry['latency_ms']:.0f}ms"
                )
            
            console.print(table)
            console.print()
        
        else:
            console.print("[red]Invalid choice[/red]")
            console.print()
