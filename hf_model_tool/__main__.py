#!/usr/bin/env python3
"""
HF-MODEL-TOOL: HuggingFace Model Management Tool

A CLI tool for managing locally downloaded HuggingFace models and datasets
"""
import sys
import logging
from typing import NoReturn
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
        features.append("Asset Listing", style="white")
        features.append(" - View models & datasets with size info\n", style="dim white")
        features.append("  • ", style="cyan")
        features.append("Duplicate Detection", style="white")
        features.append(" - Find and clean duplicate downloads\n", style="dim white")
        features.append("  • ", style="cyan")
        features.append("Asset Details", style="white")
        features.append(" - View model configs and dataset info\n", style="dim white")

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
        help_text.append("  '", style="dim white")
        help_text.append("Main Menu", style="cyan")
        help_text.append("' and '", style="dim white")
        help_text.append("Exit", style="cyan")
        help_text.append("' available everywhere", style="dim white")

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
