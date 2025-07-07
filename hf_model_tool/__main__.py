#!/usr/bin/env python3
"""
HF-MODEL-TOOL: HuggingFace Model Management Tool

A CLI tool for managing locally downloaded HuggingFace models and datasets
"""
import sys
import logging
import argparse
from typing import NoReturn, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

from . import __version__
from .cache import scan_all_directories
from .ui import (
    print_items,
    delete_assets_workflow,
    deduplicate_assets_workflow,
    view_asset_details_workflow,
)
from .navigation import unified_prompt
from .config import ConfigManager

# Configure logging - only to file, not console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path.home() / ".hf-model-tool.log")
        # Removed StreamHandler to stop console logging
    ],
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hf-model-tool",
        description="HuggingFace Model Management Tool - Organize your local AI assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hf-model-tool                           # Interactive mode
  hf-model-tool -l                        # List all assets
  hf-model-tool -m                        # Manage assets
  hf-model-tool -v                        # View asset details
  hf-model-tool -path ~/my-lora-models    # Add LoRA adapter directory
  hf-model-tool -path /data/custom-models # Add custom model directory
  hf-model-tool -l --sort name            # List assets sorted by name
        """,
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all assets with sizes",
    )
    
    parser.add_argument(
        "-m", "--manage",
        action="store_true",
        help="Enter asset management mode",
    )
    
    parser.add_argument(
        "-v", "--view", "--details",
        action="store_true",
        dest="details",
        help="View asset details",
    )
    
    
    parser.add_argument(
        "-path", "--add-path",
        type=str,
        metavar="PATH",
        help="Add a directory containing AI assets (HuggingFace cache, LoRA adapters, custom models, etc.)",
    )
    
    parser.add_argument(
        "--sort",
        choices=["size", "name", "date"],
        default="size",
        help="Sort assets by size, name, or date (default: size)",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"hf-model-tool {__version__}",
    )
    
    return parser


def handle_cli_list(sort_by: str = "size") -> None:
    """Handle the -l/--list CLI option."""
    console = Console()
    try:
        items = scan_all_directories()
        if not items:
            console.print("[yellow]No HuggingFace assets found![/yellow]")
            console.print("Use -path to add directories to scan.")
            return
        
        print_items(items, sort_by=sort_by)
    except Exception as e:
        console.print(f"[red]Error listing assets: {e}[/red]")
        logger.error(f"Error in CLI list: {e}")


def handle_cli_manage() -> None:
    """Handle the -m/--manage CLI option."""
    console = Console()
    try:
        items = scan_all_directories()
        if not items:
            console.print("[yellow]No HuggingFace assets found![/yellow]")
            console.print("Use -path to add directories to scan.")
            return
        
        while True:
            manage_choice = unified_prompt(
                "manage_action",
                "Asset Management Options",
                ["Delete Assets...", "Deduplicate Assets"],
                allow_back=True,
            )
            if not manage_choice or manage_choice == "BACK":
                break
            elif manage_choice == "MAIN_MENU":
                break
            
            if manage_choice == "Delete Assets...":
                delete_assets_workflow(items)
                break
            elif manage_choice == "Deduplicate Assets":
                deduplicate_assets_workflow(items)
                break
                
    except Exception as e:
        console.print(f"[red]Error in manage mode: {e}[/red]")
        logger.error(f"Error in CLI manage: {e}")


def handle_cli_details() -> None:
    """Handle the -d/--details CLI option."""
    console = Console()
    try:
        items = scan_all_directories()
        if not items:
            console.print("[yellow]No HuggingFace assets found![/yellow]")
            console.print("Use -path to add directories to scan.")
            return
        
        view_asset_details_workflow(items)
    except Exception as e:
        console.print(f"[red]Error viewing asset details: {e}[/red]")
        logger.error(f"Error in CLI details: {e}")


def handle_cli_add_path(path: str) -> None:
    """Handle the -path/--add-path CLI option."""
    console = Console()
    try:
        # Expand user path
        expanded_path = Path(path).expanduser().resolve()
        
        # Check if path exists
        if not expanded_path.exists():
            console.print(f"[red]Error: Path '{path}' does not exist[/red]")
            return
        
        if not expanded_path.is_dir():
            console.print(f"[red]Error: Path '{path}' is not a directory[/red]")
            return
        
        # Ask user for path type
        console.print(f"\n[bold]Adding directory:[/bold] {expanded_path}")
        console.print("\n[bold]Select Directory Type:[/bold]")
        console.print("[cyan]1. HuggingFace Cache[/cyan] - Standard HF cache with models--publisher--name structure")
        console.print("[cyan]2. Custom Directory[/cyan] - LoRA adapters, fine-tuned models, or other custom formats")
        console.print("[cyan]3. Auto-detect[/cyan] - Let the tool determine the type automatically")
        
        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == "1":
                path_type = "huggingface_cache"
                break
            elif choice == "2":
                path_type = "custom_directory"
                break
            elif choice == "3":
                path_type = "auto"
                break
            else:
                console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")
        
        # Add the directory
        config = ConfigManager()
        if config.add_directory(str(expanded_path), path_type):
            console.print(f"[green]✓ Successfully added directory: {expanded_path}[/green]")
            console.print(f"[green]  Path type: {path_type}[/green]")
        else:
            console.print(f"[yellow]Directory already exists in configuration[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error adding path: {e}[/red]")
        logger.error(f"Error in CLI add path: {e}")


def show_welcome_screen() -> None:
    """
    Display a welcome screen with ASCII art and system info

    Shows the HF-MODEL-TOOL logo, system status, feature overview,
    and navigation help in a professionally formatted layout.
    """
    console = Console()
    logger.info("Displaying welcome screen")

    try:
        # ASCII art logo
        logo = """
██   ██ ███████       ███    ███  ██████  ██████  ███████ ██           ████████  ██████   ██████  ██
██   ██ ██            ████  ████ ██    ██ ██   ██ ██      ██              ██    ██    ██ ██    ██ ██
███████ █████   █████ ██ ████ ██ ██    ██ ██   ██ █████   ██      █████   ██    ██    ██ ██    ██ ██
██   ██ ██            ██  ██  ██ ██    ██ ██   ██ ██      ██              ██    ██    ██ ██    ██ ██
██   ██ ██            ██      ██  ██████  ██████  ███████ ███████         ██     ██████   ██████  ███████
"""

        # Create colored logo
        logo_text = Text(logo, style="bold cyan")

        # Subtitle and version
        subtitle = Text(
            "🤗 HuggingFace Model Management Tool",
            style="bold yellow",
            justify="center",
        )
        version_text = Text(f"v{__version__}", style="dim white", justify="center")
        tagline = Text(
            "Organize • Clean • Optimize Your Local AI Assets",
            style="italic green",
            justify="center",
        )

        # System info with error handling
        # Quick scan for asset count across all configured directories
        try:
            items = scan_all_directories()
            if items:
                total_size = (
                    sum(item["size"] for item in items if isinstance(item["size"], int))
                    / 1e9
                )  # Convert to GB
                asset_count = len(items)

                # Count unique directories
                unique_dirs = set(item.get("source_dir", "") for item in items)
                dir_count = len(unique_dirs)

                if dir_count > 1:
                    status = f"✅ Found {asset_count} assets using {total_size:.1f} GB across {dir_count} directories"
                else:
                    status = f"✅ Found {asset_count} assets using {total_size:.1f} GB"
                status_style = "bold green"
                logger.info(
                    f"Cache scan successful: {asset_count} assets, {total_size:.1f} GB"
                )
            else:
                status = "⚠️  No HuggingFace assets found in configured directories"
                status_style = "bold yellow"
                logger.info("No assets found in any configured directory")
        except Exception as e:
            status = "⚠️  Failed to scan directories"
            status_style = "bold yellow"
            logger.warning(f"Directory scan failed: {e}")

        # Features list
        features = Text()
        features.append("🎯 Features:\n", style="bold white")
        features.append("  • ", style="cyan")
        features.append("Smart Asset Detection", style="white")
        features.append(" - LLM, LoRA Adapters, and Datasets\n", style="dim white")
        features.append("  • ", style="cyan")
        features.append("Asset Management", style="white")
        features.append(" - List, view details, and clean duplicates\n", style="dim white")
        features.append("  • ", style="cyan")
        features.append("Multi-Directory Support", style="white")
        features.append(" - Scan HuggingFace cache and custom directories\n", style="dim white")


        # Quick help
        help_text = Text()
        help_text.append("🚀 Quick Start:\n", style="bold white")
        help_text.append(
            "  Navigate with ↑/↓ arrows • Press Enter to select\n", style="dim white"
        )
        help_text.append("  Use '", style="dim white")
        help_text.append("← Back", style="cyan")
        help_text.append("' and '", style="dim white")
        help_text.append("→ Config", style="cyan")
        help_text.append("' for navigation\n", style="dim white")
        help_text.append("  Add directories via Config > Manage Directories\n", style="dim white")

        # Display the welcome screen with centered logo
        centered_logo = Align.center(logo_text)
        console.print(Panel(centered_logo, border_style="bright_blue", padding=(1, 2)))
        console.print(Align.center(subtitle))
        console.print(Align.center(version_text))
        console.print(Align.center(tagline))
        console.print("")

        # Status info
        status_text = Text(status, style=status_style)
        console.print(Align.center(status_text))
        console.print("")

        # Features and help
        console.print("")
        columns = Columns(
            [
                Panel(
                    features,
                    title="[bold cyan]Features[/bold cyan]",
                    border_style="cyan",
                ),
                Panel(
                    help_text,
                    title="[bold green]Navigation[/bold green]",
                    border_style="green",
                ),
            ],
            equal=True,
            expand=True,
        )
        console.print(columns)

        console.print("")
        console.print(
            Panel(
                "[bold white]Press Enter to continue...[/bold white]",
                style="dim",
                border_style="dim",
            )
        )

        # Wait for user input
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            logger.info("User interrupted welcome screen")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error displaying welcome screen: {e}")
        console.print(f"[red]Error displaying welcome screen: {e}[/red]")
        console.print("[yellow]Continuing to main menu...[/yellow]")


def main() -> NoReturn:
    """
    Main application entry point.

    Manages the primary application loop, handles user interactions,
    and coordinates between different workflows.
    """
    logger.info("Starting HF-MODEL-TOOL application")

    try:
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Handle CLI arguments (non-interactive mode)
        if args.list:
            handle_cli_list(args.sort)
            return
        
        if args.manage:
            handle_cli_manage()
            return
        
        if args.details:
            handle_cli_details()
            return
        
        if args.add_path:
            handle_cli_add_path(args.add_path)
            return
        
        # If no CLI arguments, run interactive mode
        # Show welcome screen on first run
        show_welcome_screen()

        while True:
            try:
                action = unified_prompt(
                    "action",
                    "Main Menu",
                    ["List Assets", "Manage Assets...", "View Asset Details", "Quit"],
                    allow_back=False,
                )

                if not action or action == "Quit":
                    # Show goodbye message
                    console = Console()
                    console.print("")
                    console.print(
                        Panel(
                            "[bold cyan]Thanks for using HF-MODEL-TOOL![/bold cyan]\n"
                            + "[dim white]Keep your AI assets organized! 🤗[/dim white]",
                            style="dim",
                            border_style="blue",
                        )
                    )
                    logger.info("User quit application")
                    break

                # Handle special navigation returns
                if action == "MAIN_MENU":
                    continue  # Stay in main menu loop

                # Handle sort options returned from config
                if action and action.startswith("Sort Assets"):
                    sort_by = "size"
                    if "Date" in action:
                        sort_by = "date"
                    elif "Name" in action:
                        sort_by = "name"

                    logger.info(f"Listing assets sorted by {sort_by}")
                    items = scan_all_directories()
                    print_items(items, sort_by=sort_by)
                    continue

                # Get items for main workflows
                items = scan_all_directories()
                logger.info(f"Loaded {len(items)} items from all directories")

                # Check if no items found
                if not items:
                    console = Console()
                    console.print("\n[yellow]No HuggingFace assets found![/yellow]")
                    console.print(
                        "\nYou can add directories containing HuggingFace assets by:"
                    )
                    console.print("  1. Go to → Config → Manage Cache Directories")
                    console.print(
                        "  2. Add directories with your downloaded models/datasets"
                    )
                    console.print("\n[dim]Press Enter to continue...[/dim]")
                    input()
                    continue

                if action == "List Assets":
                    # Default to size sorting, but user can change via config
                    logger.info("Displaying asset list")
                    print_items(items, sort_by="size")

                elif action == "Manage Assets...":
                    logger.info("Entering asset management workflow")
                    while True:  # Manage submenu loop
                        manage_choice = unified_prompt(
                            "manage_action",
                            "Asset Management Options",
                            ["Delete Assets...", "Deduplicate Assets"],
                            allow_back=True,
                        )
                        if not manage_choice or manage_choice == "BACK":
                            break  # Back to main menu
                        elif manage_choice == "MAIN_MENU":
                            break  # Back to main menu

                        if manage_choice == "Delete Assets...":
                            logger.info("Starting delete assets workflow")
                            result = delete_assets_workflow(items)
                            if result == "MAIN_MENU":
                                break  # Back to main menu
                        elif manage_choice == "Deduplicate Assets":
                            logger.info("Starting deduplicate assets workflow")
                            result = deduplicate_assets_workflow(items)
                            if result == "MAIN_MENU":
                                break  # Back to main menu

                elif action == "View Asset Details":
                    logger.info("Starting view asset details workflow")
                    result = view_asset_details_workflow(items)
                    if result == "MAIN_MENU":
                        continue  # Back to main menu

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                console = Console()
                console.print(f"[red]Error: {e}[/red]")
                console.print("[yellow]Returning to main menu...[/yellow]")
                continue

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        console = Console()
        console.print("\n[yellow]Application interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console = Console()
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)
    finally:
        logger.info("Application terminated")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
