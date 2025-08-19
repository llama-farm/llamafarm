#!/usr/bin/env python3
"""
Demo: Image Recognition System (CLI-based)

This demo showcases the image recognition capabilities of the LlamaFarm models framework
using CLI commands. All functionality is demonstrated through the CLI interface.

The demo shows:
- Hardware detection (Apple Silicon MPS, NVIDIA CUDA, CPU)
- Object detection with YOLO models
- Batch processing capabilities
- Performance benchmarking
- Multiple output formats
- Cross-platform support

NO LOGIC IN THIS FILE - just CLI commands!
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os

# Setup rich console for beautiful output
console = Console()

# Change to models directory for CLI execution
MODELS_DIR = Path(__file__).parent.parent
os.chdir(MODELS_DIR)

def run_cli_command(command: str, description: str = None) -> tuple[bool, str]:
    """Run a CLI command and return success status and output."""
    if description:
        console.print(f"\n[bold cyan]→ {description}[/bold cyan]")
    
    console.print(f"[dim]$ {command}[/dim]")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=MODELS_DIR
        )
        
        # Print output with formatting
        if result.stdout:
            for line in result.stdout.split('\n'):
                if '✅' in line or 'Success' in line:
                    console.print(f"[green]{line}[/green]")
                elif '❌' in line or 'ERROR' in line:
                    console.print(f"[red]{line}[/red]")
                elif '🔍' in line or '📸' in line or '🎯' in line:
                    console.print(f"[yellow]{line}[/yellow]")
                elif 'Device:' in line or 'Hardware:' in line:
                    console.print(f"[bold magenta]{line}[/bold magenta]")
                elif 'FPS' in line or 'ms' in line:
                    console.print(f"[bold blue]{line}[/bold blue]")
                elif any(word in line.lower() for word in ['person', 'car', 'dog', 'cat', 'bird']):
                    console.print(f"[bold green]{line}[/bold green]")
                else:
                    console.print(line)
        
        # Only show stderr if it contains actual errors (not INFO or WARNING)
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            error_lines = []
            for line in stderr_lines:
                # Skip INFO, WARNING, progress bars, and processing messages
                if (line and 
                    ' - INFO - ' not in line and
                    ' - WARNING - ' not in line and
                    ' - DEBUG - ' not in line and
                    'WARNING' not in line and
                    '%' not in line and
                    '█' not in line and
                    'Processing' not in line and
                    'Extracting' not in line and
                    'Embedding' not in line and
                    'Adding' not in line):
                    # Only keep actual ERROR or CRITICAL messages (skip known non-critical errors)
                    if (' - ERROR - ' in line or ' - CRITICAL - ' in line or 'Error:' in line or 'error' in line.lower()) and \
                       'tostring_rgb' not in line:  # Skip matplotlib backend warning
                        error_lines.append(line)
            
            if error_lines:
                console.print(f"[red]Error: {' '.join(error_lines)}[/red]")
            
        return result.returncode == 0, result.stdout
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")
        return False, str(e)


def print_section_header(title: str, emoji: str = "📸"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def wait_for_enter(message: str = "Press Enter to continue..."):
    """Wait for user to press Enter."""
    console.print(f"\n[dim]{message}[/dim]")
    input()


def can_display_images_inline():
    """Check if terminal supports inline image display."""
    # Check for iTerm2
    if os.environ.get('TERM_PROGRAM') == 'iTerm.app':
        return 'iterm'
    # Check for Kitty
    if os.environ.get('TERM') == 'xterm-kitty':
        return 'kitty'
    # Check for WezTerm
    if 'WezTerm' in os.environ.get('TERM_PROGRAM', ''):
        return 'wezterm'
    # Check for VSCode terminal
    if os.environ.get('TERM_PROGRAM') == 'vscode':
        return 'vscode'
    return None


def display_image_inline(image_path: Path, terminal_type: str):
    """Display image inline if terminal supports it."""
    if terminal_type == 'iterm':
        # iTerm2 inline images protocol
        try:
            import base64
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            print(f'\033]1337;File=inline=1;width=50%:{img_data}\a')
            return True
        except:
            pass
    # Add other terminal protocols as needed
    return False


def demonstrate_image_recognition_cli():
    """Demonstrate image recognition using CLI commands only."""
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]🦙 LlamaFarm Image Recognition Demo[/bold cyan]\n"
        "[yellow]Cross-platform object detection with YOLO[/yellow]",
        border_style="cyan"
    ))
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]100% CLI-based operation[/bold cyan] - no hardcoded logic!")
    console.print("• Hardware acceleration (MPS, CUDA, CPU)")
    console.print("• YOLO object detection models")
    console.print("• Batch processing capabilities")
    console.print("• Performance benchmarking")
    console.print("• Multiple output formats")
    console.print("\n[dim]All processing through the CLI interface[/dim]")
    
    wait_for_enter()
    
    # Step 1: Check system information
    print_section_header("System Information", "🖥️")
    
    run_cli_command(
        "uv run python cli.py image info --device",
        "Detecting available hardware acceleration"
    )
    
    time.sleep(2)
    
    run_cli_command(
        "uv run python cli.py image info --models",
        "Listing available YOLO models"
    )
    
    wait_for_enter()
    
    # Step 2: Setup verification
    print_section_header("Setup Verification", "✔️")
    
    run_cli_command(
        "uv run python cli.py image setup --test",
        "Verifying image recognition system is ready"
    )
    
    wait_for_enter()
    
    # Step 3: Object detection on sample images
    print_section_header("Object Detection", "🎯")
    
    # Check if sample images exist
    sample_dir = Path("demos/sample_images")
    if not sample_dir.exists():
        console.print("[yellow]Sample images not found. Downloading...[/yellow]")
        run_cli_command(
            "uv run python cli.py image download-sample --output-dir demos/sample_images --type all",
            "Downloading sample images"
        )
        time.sleep(2)
    
    # Detect objects in street scene
    console.print("\n[bold yellow]Detection 1: Street Scene[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --confidence 0.5 --measure-time --output-format summary",
        "Detecting objects in street scene (buses, people, cars) using default YOLO strategy"
    )
    
    time.sleep(2)
    
    # Detect objects in people image
    console.print("\n[bold yellow]Detection 2: People in Stadium[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/people.jpg --strategy image_yolo_default --confidence 0.5 --output-format summary",
        "Detecting people in stadium image using default YOLO strategy"
    )
    
    time.sleep(2)
    
    # Detect objects in airplane image if exists
    airplane_path = Path("demos/sample_images/airplane_sky.jpg")
    if airplane_path.exists():
        console.print("\n[bold yellow]Detection 3: Airplane in Sky[/bold yellow]")
        run_cli_command(
            "uv run python cli.py image detect demos/sample_images/airplane_sky.jpg --strategy image_yolo_default --confidence 0.4 --output-format summary",
            "Detecting airplane and sky objects using default YOLO strategy"
        )
        time.sleep(2)
    
    # Detect objects in dog image if exists
    dog_path = Path("demos/sample_images/dog.jpg")
    if dog_path.exists():
        console.print("\n[bold yellow]Detection 4: Animals[/bold yellow]")
        run_cli_command(
            "uv run python cli.py image detect demos/sample_images/dog.jpg --strategy image_yolo_default --confidence 0.4 --output-format summary",
            "Detecting animals in image using default YOLO strategy"
        )
    
    wait_for_enter()
    
    # Step 4: Different output formats
    print_section_header("Output Formats", "📊")
    
    console.print("\n[bold yellow]JSON Format (for API integration)[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --confidence 0.5 --output-format json | head -20",
        "JSON output for programmatic use"
    )
    
    time.sleep(2)
    
    console.print("\n[bold yellow]CSV Format (for data analysis)[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --confidence 0.5 --output-format csv",
        "CSV output for spreadsheets"
    )
    
    wait_for_enter()
    
    # Step 5: Batch processing
    print_section_header("Batch Processing", "🚀")
    
    # Ensure output directory exists
    Path("demos/demo_outputs_vision/batch_results").mkdir(parents=True, exist_ok=True)
    
    console.print("[yellow]Processing all images in demos/sample_images/:[/yellow]")
    console.print("  • street_scene.jpg")
    console.print("  • people.jpg")
    
    run_cli_command(
        "uv run python cli.py image batch-detect demos/sample_images --output-dir demos/demo_outputs_vision/batch_results --summary",
        "Batch processing both images in parallel"
    )
    
    wait_for_enter()
    
    # Step 6: Performance benchmark
    print_section_header("Performance Benchmark", "⚡")
    
    run_cli_command(
        "uv run python cli.py image benchmark --runs 5",
        "Benchmarking detection speed on your hardware"
    )
    
    wait_for_enter()
    
    # Step 7: Detection with visualization
    print_section_header("Visual Detection Results", "🖼️")
    
    # Create output directory if it doesn't exist
    Path("demos/demo_outputs_vision").mkdir(exist_ok=True)
    
    console.print("\n[bold yellow]Visualization 1: Street Scene with Bounding Boxes[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --visualize --output-path demos/demo_outputs_vision/street_scene_detected.jpg",
        "Creating annotated street scene image using default strategy"
    )
    
    time.sleep(2)
    
    console.print("\n[bold yellow]Visualization 2: People Detection with Bounding Boxes[/bold yellow]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/people.jpg --strategy image_yolo_default --visualize --output-path demos/demo_outputs_vision/people_detected.jpg",
        "Creating annotated people image using default strategy"
    )
    
    time.sleep(2)
    
    # Add airplane detection visualization if exists
    airplane_path = Path("demos/sample_images/airplane_sky.jpg")
    if airplane_path.exists():
        console.print("\n[bold yellow]Visualization 3: Airplane Detection with Bounding Boxes[/bold yellow]")
        run_cli_command(
            "uv run python cli.py image detect demos/sample_images/airplane_sky.jpg --strategy image_yolo_default --visualize --output-path demos/demo_outputs_vision/airplane_detected.jpg",
            "Creating annotated airplane image using default strategy"
        )
        time.sleep(2)
    
    # Add dog detection visualization if exists
    dog_path = Path("demos/sample_images/dog.jpg")
    if dog_path.exists():
        console.print("\n[bold yellow]Visualization 4: Dog Detection with Bounding Boxes[/bold yellow]")
        run_cli_command(
            "uv run python cli.py image detect demos/sample_images/dog.jpg --strategy image_yolo_default --visualize --output-path demos/demo_outputs_vision/dog_detected.jpg",
            "Creating annotated dog image using default strategy"
        )
        time.sleep(2)
    
    console.print("\n[green]✅ Annotated images saved to demos/demo_outputs_vision/[/green]")
    console.print("  • street_scene_detected.jpg")
    console.print("  • people_detected.jpg")
    if airplane_path.exists():
        console.print("  • airplane_detected.jpg")
    if dog_path.exists():
        console.print("  • dog_detected.jpg")
    
    # Make it interactive - offer to open the images
    console.print("\n[cyan]View the annotated images:[/cyan]")
    output_path_1 = Path("demos/demo_outputs_vision/street_scene_detected.jpg").absolute()
    output_path_2 = Path("demos/demo_outputs_vision/people_detected.jpg").absolute()
    output_path_3 = Path("demos/demo_outputs_vision/airplane_detected.jpg").absolute()
    output_path_4 = Path("demos/demo_outputs_vision/dog_detected.jpg").absolute()
    
    # Check if terminal supports inline images
    terminal_type = can_display_images_inline()
    if terminal_type:
        console.print(f"[green]✨ Your terminal ({terminal_type}) supports inline images![/green]")
        if output_path_1.exists():
            console.print("\n[yellow]Street Scene with Detections:[/yellow]")
            display_image_inline(output_path_1, terminal_type)
        if output_path_2.exists():
            console.print("\n[yellow]People with Detections:[/yellow]")
            display_image_inline(output_path_2, terminal_type)
        if output_path_3.exists():
            console.print("\n[yellow]Airplane with Detections:[/yellow]")
            display_image_inline(output_path_3, terminal_type)
        if output_path_4.exists():
            console.print("\n[yellow]Dog with Detections:[/yellow]")
            display_image_inline(output_path_4, terminal_type)
    else:
        # Show clickable file URLs (works in many modern terminals)
        console.print(f"  📁 [link=file://{output_path_1}]file://{output_path_1}[/link]")
        console.print(f"  📁 [link=file://{output_path_2}]file://{output_path_2}[/link]")
        if output_path_3.exists():
            console.print(f"  📁 [link=file://{output_path_3}]file://{output_path_3}[/link]")
        if output_path_4.exists():
            console.print(f"  📁 [link=file://{output_path_4}]file://{output_path_4}[/link]")
        console.print("\n[dim]💡 Tip: Click the links above (if your terminal supports it)[/dim]")
    
    # Always offer to open in external viewer
    console.print("\n[yellow]Interactive Options:[/yellow]")
    console.print("  [bold]1[/bold] - Open street scene in image viewer")
    console.print("  [bold]2[/bold] - Open people image in image viewer")
    if output_path_3.exists():
        console.print("  [bold]3[/bold] - Open airplane image in image viewer")
    if output_path_4.exists():
        console.print("  [bold]4[/bold] - Open dog image in image viewer")
    console.print("  [bold]a[/bold] - Open all images")
    console.print("  [bold]s[/bold] - Show detection statistics")
    console.print("  [bold]Enter[/bold] - Continue demo")
    
    response = console.input("\n[cyan]Choose an option: [/cyan]").strip().lower()
    
    if response == '1':
        import webbrowser
        webbrowser.open(f"file://{output_path_1}")
        console.print("[green]✅ Opened street scene in your default viewer[/green]")
        wait_for_enter("Press Enter after viewing the image...")
    elif response == '2':
        import webbrowser
        webbrowser.open(f"file://{output_path_2}")
        console.print("[green]✅ Opened people image in your default viewer[/green]")
        wait_for_enter("Press Enter after viewing the image...")
    elif response == '3' and output_path_3.exists():
        import webbrowser
        webbrowser.open(f"file://{output_path_3}")
        console.print("[green]✅ Opened airplane image in your default viewer[/green]")
        wait_for_enter("Press Enter after viewing the image...")
    elif response == '4' and output_path_4.exists():
        import webbrowser
        webbrowser.open(f"file://{output_path_4}")
        console.print("[green]✅ Opened dog image in your default viewer[/green]")
        wait_for_enter("Press Enter after viewing the image...")
    elif response == 'a':
        import webbrowser
        webbrowser.open(f"file://{output_path_1}")
        webbrowser.open(f"file://{output_path_2}")
        if output_path_3.exists():
            webbrowser.open(f"file://{output_path_3}")
        if output_path_4.exists():
            webbrowser.open(f"file://{output_path_4}")
        console.print("[green]✅ Opened all images in your default viewer[/green]")
        wait_for_enter("Press Enter after viewing the images...")
    elif response == 's':
        console.print("\n[bold cyan]Detection Statistics:[/bold cyan]")
        # Run actual detection commands to get real statistics
        console.print("[dim]Running detection analysis...[/dim]\n")
        
        # Get actual stats for street scene
        result1 = subprocess.run(
            "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --output-format json",
            shell=True, capture_output=True, text=True, cwd=MODELS_DIR
        )
        
        # Get actual stats for people image
        result2 = subprocess.run(
            "uv run python cli.py image detect demos/sample_images/people.jpg --strategy image_yolo_default --output-format json",
            shell=True, capture_output=True, text=True, cwd=MODELS_DIR
        )
        
        # Get actual stats for airplane image if exists
        all_detections = []
        airplane_path = Path("demos/sample_images/airplane_sky.jpg")
        if airplane_path.exists():
            result3 = subprocess.run(
                "uv run python cli.py image detect demos/sample_images/airplane_sky.jpg --strategy image_yolo_default --output-format json",
                shell=True, capture_output=True, text=True, cwd=MODELS_DIR
            )
        else:
            result3 = None
        
        # Get actual stats for dog image if exists
        dog_path = Path("demos/sample_images/dog.jpg")
        if dog_path.exists():
            result4 = subprocess.run(
                "uv run python cli.py image detect demos/sample_images/dog.jpg --strategy image_yolo_default --output-format json",
                shell=True, capture_output=True, text=True, cwd=MODELS_DIR
            )
        else:
            result4 = None
        
        # Parse and display results
        try:
            import json
            if result1.returncode == 0:
                detections1 = json.loads(result1.stdout)
                console.print("[yellow]Street Scene Results:[/yellow]")
                obj_counts = {}
                for det in detections1:
                    label = det.get('label', 'unknown')
                    obj_counts[label] = obj_counts.get(label, 0) + 1
                for label, count in obj_counts.items():
                    console.print(f"  - {count} {label}(s)")
                console.print(f"  Total: {len(detections1)} objects\n")
                all_detections.extend(detections1)
            
            if result2.returncode == 0:
                detections2 = json.loads(result2.stdout)
                console.print("[yellow]People Image Results:[/yellow]")
                obj_counts = {}
                for det in detections2:
                    label = det.get('label', 'unknown')
                    obj_counts[label] = obj_counts.get(label, 0) + 1
                for label, count in obj_counts.items():
                    console.print(f"  - {count} {label}(s)")
                console.print(f"  Total: {len(detections2)} objects\n")
                all_detections.extend(detections2)
            
            if result3 and result3.returncode == 0:
                detections3 = json.loads(result3.stdout)
                console.print("[yellow]Airplane Image Results:[/yellow]")
                obj_counts = {}
                for det in detections3:
                    label = det.get('label', 'unknown')
                    obj_counts[label] = obj_counts.get(label, 0) + 1
                for label, count in obj_counts.items():
                    console.print(f"  - {count} {label}(s)")
                console.print(f"  Total: {len(detections3)} objects\n")
                all_detections.extend(detections3)
            
            if result4 and result4.returncode == 0:
                detections4 = json.loads(result4.stdout)
                console.print("[yellow]Dog Image Results:[/yellow]")
                obj_counts = {}
                for det in detections4:
                    label = det.get('label', 'unknown')
                    obj_counts[label] = obj_counts.get(label, 0) + 1
                for label, count in obj_counts.items():
                    console.print(f"  - {count} {label}(s)")
                console.print(f"  Total: {len(detections4)} objects\n")
                all_detections.extend(detections4)
            
            # Overall stats
            if all_detections:
                console.print("[green]Overall Statistics:[/green]")
                console.print(f"  Total objects detected: {len(all_detections)}")
                console.print(f"  Average confidence: {sum(d['confidence'] for d in all_detections) / len(all_detections):.2%}")
        except:
            console.print("[red]Could not parse detection results[/red]")
        
        wait_for_enter()
    
    wait_for_enter()
    
    # Step 8: Strategy comparison
    print_section_header("Strategy Comparison", "📈")
    
    console.print("\n[bold]Available Image Recognition Strategies:[/bold]")
    
    # Show available strategies
    run_cli_command(
        "uv run python cli.py image info --strategies",
        "Displaying available image recognition strategies"
    )
    
    console.print("\n[bold yellow]Strategy Demonstration: Comparing default vs performance[/bold yellow]")
    
    # Demonstrate performance strategy
    console.print("\n[cyan]Using high-performance strategy (yolov8s):[/cyan]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_performance --measure-time --output-format summary",
        "Detection with performance-optimized strategy"
    )
    
    time.sleep(2)
    
    # Demonstrate accuracy strategy 
    console.print("\n[cyan]Using high-accuracy strategy (yolov8l):[/cyan]")
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_accuracy --measure-time --output-format summary",
        "Detection with accuracy-optimized strategy"
    )
    
    wait_for_enter()
    
    # Step 9: Advanced features
    print_section_header("Advanced Features", "🎨")
    
    console.print("\n[bold]Additional CLI capabilities:[/bold]")
    console.print("• [cyan]Classification[/cyan]: Identify image categories")
    console.print("• [cyan]Segmentation[/cyan]: Pixel-level object detection")
    console.print("• [cyan]Few-shot training[/cyan]: Train custom detectors")
    console.print("• [cyan]Model export[/cyan]: ONNX, CoreML, TensorFlow Lite")
    console.print("• [cyan]URL detection[/cyan]: Process images from web")
    
    console.print("\n[bold]Example commands:[/bold]")
    
    command_table = Table(show_header=True, header_style="bold magenta")
    command_table.add_column("Feature", style="cyan")
    command_table.add_column("Command", style="white")
    
    commands = [
        ("Classify image", "cli.py image classify image.jpg --top-k 5"),
        ("Segment objects", "cli.py image segment image.jpg --output mask.png"),
        ("Train detector", "cli.py image train dataset/ --epochs 10"),
        ("Export model", "cli.py image export model.pt --format onnx"),
        ("Detect from URL", "cli.py image detect https://example.com/image.jpg")
    ]
    
    for feature, cmd in commands:
        command_table.add_row(feature, cmd)
    
    console.print(command_table)
    
    # Step 10: Hardware optimization
    print_section_header("Hardware Optimization", "🔧")
    
    console.print("\n[bold]Performance by hardware:[/bold]")
    hw_table = Table(show_header=False, show_edge=False)
    hw_table.add_column("", style="yellow", width=25)
    hw_table.add_column("", style="white")
    
    hardware_info = [
        ("Apple Silicon (MPS)", "28+ FPS with M1/M2/M3"),
        ("NVIDIA GPU (CUDA)", "40+ FPS with RTX series"),
        ("CPU Only", "5-10 FPS depending on model"),
        ("Optimization", "Auto-detects best device")
    ]
    
    for hw, performance in hardware_info:
        hw_table.add_row(hw, performance)
    
    console.print(hw_table)
    
    # Summary
    print_section_header("Demo Summary", "🎓")
    
    summary_points = [
        ("CLI-driven", "All features accessible via command line"),
        ("Cross-platform", "Works on Mac, Windows, Linux"),
        ("Hardware optimized", "Auto-detects MPS/CUDA/CPU"),
        ("Production-ready", "Batch processing, multiple formats"),
        ("Extensible", "Easy to add new models and features")
    ]
    
    summary_table = Table(show_header=False, show_edge=False)
    summary_table.add_column("", style="bold green", width=20)
    summary_table.add_column("", style="white")
    
    for point, description in summary_points:
        summary_table.add_row(f"✅ {point}", description)
    
    console.print(summary_table)
    
    console.print("\n[bold cyan]Image recognition system ready for production use![/bold cyan]")
    console.print(f"[dim]To detect objects in your own images:[/dim]")
    console.print('[dim]$ uv run python cli.py image detect your_image.jpg[/dim]')
    
    # Integration examples
    print_section_header("Integration Examples", "🔗")
    
    console.print("\n[bold]Integrate with your applications:[/bold]")
    console.print("• Python API: Use ImageRecognizerFactory directly")
    console.print("• REST API: Wrap CLI commands in web service")
    console.print("• Batch jobs: Schedule with cron/airflow")
    console.print("• CI/CD: Automated visual testing")
    console.print("• Edge deployment: Export to ONNX/CoreML")
    
    # Final demo with verbose mode
    print_section_header("Complete Detection with Verbose Mode", "🎬")
    
    console.print("[yellow]Running detection with full verbose output to show all details:[/yellow]\n")
    
    run_cli_command(
        "uv run python cli.py image detect demos/sample_images/street_scene.jpg --strategy image_yolo_default --verbose --confidence 0.5 --measure-time",
        "Complete verbose detection showing all processing steps with strategy configuration"
    )
    
    console.print("\n[bold green]🎉 Image recognition demo complete![/bold green]")


if __name__ == "__main__":
    try:
        demonstrate_image_recognition_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Image recognition demo interrupted by user", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n\n❌ Demo failed: {str(e)}", style="red")
        console.print("Ensure UV is installed and dependencies are synced", style="dim")
        sys.exit(1)