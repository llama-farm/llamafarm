"""Progress tracking utilities with llama puns and motivational messages."""

import random
import time
from typing import List
from tqdm import tqdm
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)


class LlamaProgressTracker:
    """Progress tracker with llama puns and motivational messages."""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.llama_puns = [
            "🦙 Llama-zing progress ahead!",
            "🦙 Don't have a bad llama day!",
            "🦙 Llama tell you, this is going great!",
            "🦙 No prob-llama here!",
            "🦙 Llama-nating those documents!",
            "🦙 Drama? Nah, just llama!",
            "🦙 Llama make this quick!",
            "🦙 Feeling llama-tastic!",
            "🦙 Llama get this done!",
            "🦙 Barack O-llama would be proud!",
            "🦙 Llama-geddon of productivity!",
            "🦙 Llama split these files!",
            "🦙 Como se llama? Awesome!",
            "🦙 Llama-nade stand of progress!",
            "🦙 Llama drama, just results!",
            "🦙 Dalai Llama of data processing!",
            "🦙 Llama-nificent work happening!",
            "🦙 Llama see those embeddings flow!",
            "🦙 Llama tell you a secret - we're fast!",
            "🦙 No llama left behind!",
        ]

        self.motivation_messages = [
            "Crunching through your data like a hungry llama! 🌾",
            "Your documents are getting the VIP treatment! ⭐",
            "Building your knowledge fortress, brick by brick! 🏰",
            "Transforming chaos into searchable wisdom! 🧙‍♂️",
            "Your future self will thank you for this! 🚀",
            "Making your data dreams come true! ✨",
            "Every document brings us closer to greatness! 📚",
            "Weaving a web of knowledge just for you! 🕸️",
            "Your patience is creating something beautiful! 🎨",
            "Data ingestion: where magic happens! 🪄",
        ]

        self.completion_messages = [
            "🎉 Llama-nificent! Your RAG system is ready to roll!",
            "🎊 No prob-llama! Mission accomplished!",
            "🥳 Llama tell you - that was incredible!",
            "🌟 Drama-free llama processing complete!",
            "🎯 Barack O-llama himself couldn't do it better!",
            "🏆 Llama-geddon of success achieved!",
            "🎈 Dalai Llama level of zen processing reached!",
            "💫 Como se llama this feeling? Pure joy!",
            "🚀 Llama blast off to search excellence!",
            "🎪 The greatest llama show on earth - complete!",
        ]

    def create_progress_bar(self, total: int, desc: str = "Processing") -> tqdm:
        """Create a beautiful progress bar with llama flair."""
        if self.quiet:
            # Return a no-op progress bar
            return tqdm(total=total, disable=True)
        
        return tqdm(
            total=total,
            desc=f"{Fore.CYAN}{desc}{Style.RESET_ALL}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            colour="green",
            dynamic_ncols=True,
            disable=self.quiet,
        )

    def get_random_pun(self) -> str:
        """Get a random llama pun."""
        return random.choice(self.llama_puns)

    def get_random_motivation(self) -> str:
        """Get a random motivational message."""
        return random.choice(self.motivation_messages)

    def get_completion_message(self) -> str:
        """Get a random completion message."""
        return random.choice(self.completion_messages)

    def print_header(self, title: str):
        """Print a fancy header."""
        if self.quiet:
            return
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{title.center(60)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}\n")

    def print_success(self, message: str):
        """Print a success message."""
        if not self.quiet:
            print(f"\n{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

    def print_info(self, message: str):
        """Print an info message."""
        if self.verbose:
            print(f"{Fore.CYAN}ℹ️  {message}{Style.RESET_ALL}")

    def print_warning(self, message: str):
        """Print a warning message."""
        if not self.quiet:
            print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

    def print_error(self, message: str):
        """Print an error message."""
        # Always print errors unless in quiet mode
        if not self.quiet:
            print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

    def print_verbose_results(self, results: list, title: str = "Results"):
        """Print detailed results when in verbose mode."""
        if not self.verbose or self.quiet:
            return
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{title.center(60)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{Fore.YELLOW}Result {i} - Raw Details:{Style.RESET_ALL}")
            if hasattr(result, '__dict__'):
                for key, value in result.__dict__.items():
                    if key.startswith('_'):
                        continue
                    # Truncate very long values for readability
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:197] + "..."
                    elif isinstance(value, dict):
                        # Show dict in a cleaner format
                        if len(str(value)) > 200:
                            value = f"{{...{len(value)} items...}}"
                    elif isinstance(value, list) and len(value) > 10:
                        value = f"[...{len(value)} items...]"
                    print(f"  {Fore.GREEN}{key}:{Style.RESET_ALL} {value}")
            else:
                print(f"  {result}")
            print()
    
    def print_llama_art(self):
        """Print ASCII llama art."""
        if self.quiet:
            return
        llama_art = f"""{Fore.MAGENTA}
        🦙 RAG Llama at your service!
⠀⠀⠀⡾⣦⡀⠀⠀⡀⠀⣰⢷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣠⠗⠛⠽⠛⠋⠉⢳⡃⢨⢧⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣰⠋⠁⠀⠀⠀⠀⠀⠀⠙⠛⢾⡈⡏⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣼⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠸⢦⡀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢈⠟⠓⠶⠞⠒⢻⣿⡏⢳⡀⠀⠀⠀⠀⢸⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⡴⢉⠀⠀⠀⠀⠀⠈⠛⢁⣸⠇⠀⠀⠀⠀⢺⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢧⣸⡁⠀⠀⣀⠀⠀⣠⠾⠀⠀⠀⠀⠀⠀⣹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠉⠓⢲⠾⣍⣀⣀⡿⠃⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⡗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣸⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠺⠦⠤⠤⣤⣄⣀⣀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠳⣦⣄⠀⠀
⠀⠀⢀⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣆⠀
⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆
⠀⠀⣏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿
⠀⠀⢹⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼
⠀⠀⠀⣏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡞
⠀⠀⠀⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇
⠀⠀⠀⠀⠈⢻⣦⣀⠀⣏⠀⠀⠀⠀⠀⠀⢸⡆⠀⠀⢠⡄⠀⠀⠀⠀⠀⢀⡿⠀
⠀⠀⠀⠀⠀⠀⠻⡉⠙⢻⡆⠀⠀⠀⠀⠀⡾⠚⠓⣖⠛⣧⡀⠀⠀⠀⢀⡾⠁⠀
⠀⠀⠀⠀⠀⠀⠀⠙⡇⢀⡿⣦⡀⠀⢀⡴⠃⠀⠀⠈⣷⢈⠷⡆⠀⣴⠛⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠛⠚⠀⢸⡇⣰⠏⠁⠀⠀⠀⠀⢉⠁⢸⠷⠼⠃⠀⠀⠀⠀
{Style.RESET_ALL}"""
        print(llama_art)


def create_enhanced_progress_bar(total: int, desc: str, tracker: LlamaProgressTracker):
    """Create an enhanced progress bar with periodic updates."""
    pbar = tracker.create_progress_bar(total, desc)

    # Track when to show motivational messages (every 20% or so)
    milestone_interval = max(1, total // 5)

    return pbar, milestone_interval
