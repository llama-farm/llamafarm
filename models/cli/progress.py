"""
Progress tracking utilities for the Models CLI.
Inspired by the RAG system's llama-themed progress tracker.
"""

import time
import random
from typing import Optional, List
from colorama import init, Fore, Style, Back
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

# Initialize colorama
init(autoreset=True)

console = Console()


class ModelProgressTracker:
    """Enhanced progress tracker for model operations with visual feedback."""
    
    # Training-themed messages
    TRAINING_MESSAGES = [
        "🦙 Fine-tuning those neurons...",
        "📚 Teaching the model new tricks...",
        "🎯 Optimizing loss functions...",
        "🔬 Adjusting learning rates...",
        "⚡ Backpropagating gradients...",
        "🎨 Crafting better representations...",
        "🧠 Expanding model knowledge...",
        "💡 Discovering patterns...",
        "🚀 Accelerating convergence...",
        "✨ Polishing parameters..."
    ]
    
    # Generation-themed messages
    GENERATION_MESSAGES = [
        "🦙 Generating response...",
        "💭 Thinking deeply...",
        "🎯 Selecting tokens...",
        "📝 Crafting output...",
        "🔮 Predicting next words...",
        "✨ Creating magic...",
        "🎨 Painting with words...",
        "🧠 Processing context...",
        "💡 Formulating ideas...",
        "🚀 Streaming tokens..."
    ]
    
    # Loading-themed messages
    LOADING_MESSAGES = [
        "🦙 Loading model weights...",
        "📦 Unpacking parameters...",
        "🔧 Initializing components...",
        "⚙️ Configuring pipeline...",
        "🎯 Setting up inference...",
        "💾 Caching embeddings...",
        "🔌 Connecting to services...",
        "🌐 Establishing connections...",
        "🚀 Preparing for launch...",
        "✨ Almost ready..."
    ]
    
    def __init__(self, verbose: bool = True):
        """Initialize the progress tracker."""
        self.verbose = verbose
        self.console = console
        
    def print_header(self, title: str, subtitle: Optional[str] = None):
        """Print a styled header."""
        if not self.verbose:
            return
            
        header_text = Text(title, style="bold cyan")
        if subtitle:
            header_text.append(f"\n{subtitle}", style="dim white")
        
        panel = Panel(
            header_text,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def print_status(self, message: str, status: str = "info"):
        """Print a status message with appropriate styling."""
        if not self.verbose:
            return
            
        styles = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "debug": "dim white"
        }
        
        style = styles.get(status, "white")
        icon = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "debug": "🔍"
        }.get(status, "•")
        
        self.console.print(f"{icon} {message}", style=style)
        
    def create_progress_bar(self, description: str = "Processing") -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True
        )
        
    def show_training_progress(self, epoch: int, total_epochs: int, 
                              loss: float, metrics: dict = None):
        """Display training progress with metrics."""
        if not self.verbose:
            return
            
        # Create a table for metrics
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Epoch", f"{epoch}/{total_epochs}")
        table.add_row("Loss", f"{loss:.4f}")
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
        
        # Add a random training message
        message = random.choice(self.TRAINING_MESSAGES)
        
        panel = Panel(
            table,
            title=f"[bold cyan]Training Progress - {message}[/bold cyan]",
            border_style="cyan"
        )
        
        self.console.print(panel)
        
    def show_generation_output(self, prompt: str, response: str, 
                              model: str = None, latency: float = None):
        """Display generation output in a styled format."""
        if not self.verbose:
            self.console.print(response)
            return
            
        # Create output panel
        output_text = Text()
        
        # Add prompt
        output_text.append("Prompt: ", style="bold cyan")
        output_text.append(f"{prompt}\n\n", style="white")
        
        # Add response
        output_text.append("Response: ", style="bold green")
        output_text.append(f"{response}\n", style="white")
        
        # Add metadata if available
        if model or latency:
            output_text.append("\n", style="white")
            if model:
                output_text.append(f"Model: {model}  ", style="dim white")
            if latency:
                output_text.append(f"Latency: {latency:.2f}s", style="dim white")
        
        panel = Panel(
            output_text,
            title="[bold green]🦙 Model Output[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
    def show_fallback_chain(self, attempts: List[dict]):
        """Display fallback chain attempts."""
        if not self.verbose:
            return
            
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Status", style="white")
        table.add_column("Reason", style="dim white")
        
        for attempt in attempts:
            status_style = "green" if attempt['success'] else "red"
            status_icon = "✅" if attempt['success'] else "❌"
            
            table.add_row(
                attempt.get('provider', 'Unknown'),
                attempt.get('model', 'Unknown'),
                f"[{status_style}]{status_icon}[/{status_style}]",
                attempt.get('reason', '')
            )
        
        panel = Panel(
            table,
            title="[bold yellow]🔄 Fallback Chain Execution[/bold yellow]",
            border_style="yellow"
        )
        
        self.console.print(panel)
        
    def show_model_comparison(self, results: List[dict]):
        """Display model comparison results."""
        if not self.verbose:
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="white")
        table.add_column("Latency", style="yellow")
        table.add_column("Tokens/s", style="green")
        table.add_column("Cost", style="red")
        
        for result in results:
            table.add_row(
                result.get('model', 'Unknown'),
                result.get('provider', 'Unknown'),
                f"{result.get('latency', 0):.2f}s",
                f"{result.get('tokens_per_sec', 0):.1f}",
                f"${result.get('cost', 0):.4f}"
            )
        
        panel = Panel(
            table,
            title="[bold magenta]📊 Model Performance Comparison[/bold magenta]",
            border_style="magenta"
        )
        
        self.console.print(panel)
        
    def print_success(self, message: str):
        """Print a success message."""
        self.print_status(message, "success")
        
    def print_error(self, message: str):
        """Print an error message."""
        self.print_status(message, "error")
        
    def print_warning(self, message: str):
        """Print a warning message."""
        self.print_status(message, "warning")
        
    def print_info(self, message: str):
        """Print an info message."""
        self.print_status(message, "info")
        
    def print_debug(self, message: str):
        """Print a debug message."""
        if self.verbose:
            self.print_status(message, "debug")


def create_spinner(text: str = "Processing...") -> Live:
    """Create a spinner for long-running operations."""
    spinner = Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{text}[/cyan]"),
        transient=True
    )
    return Live(spinner, refresh_per_second=10)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"