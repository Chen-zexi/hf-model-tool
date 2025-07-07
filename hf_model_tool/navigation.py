#!/usr/bin/env python3
"""
Unified navigation system for HF-MODEL-TOOL.

Provides consistent menu navigation, configuration management,
and help system across all application workflows.
"""
import sys
import os
import logging
from typing import Optional, List

import inquirer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

from .config import ConfigManager

logger = logging.getLogger(__name__)


def manage_directories() -> None:
    """
    Show directory management interface.

    Allows users to add, remove, and view configured directories
    for HuggingFace asset scanning.
    """
    logger.info("Entering directory management")
    config_manager = ConfigManager()
    console = Console()

    while True:
        try:
            # Get current directories
            config = config_manager.load_config()
            custom_dirs = config.get("custom_directories", [])
            include_default = config.get("include_default_cache", True)

            # Create status display
            console.print("\n[bold cyan]Cache Directory Configuration[/bold cyan]")

            # Show default cache status
            default_status = "Enabled" if include_default else "Disabled"
            console.print(f"\nDefault HuggingFace Cache: {default_status}")
            if include_default:
                console.print("  ~/.cache/huggingface/hub")
                console.print("  ~/.cache/huggingface/datasets")

            # Show custom directories
            if custom_dirs:
                console.print("\n[bold]Custom Directories:[/bold]")
                for i, dir_path in enumerate(custom_dirs, 1):
                    exists = "✅" if Path(dir_path).exists() else "❌"
                    console.print(f"  {i}. {exists} {dir_path}")
            else:
                console.print("\n[dim]No custom directories configured[/dim]")

            # Show menu options
            options = [
                "Add Directory Path",
                "Add Current Directory",
                "Remove Directory",
                "Toggle Default Cache",
                "Scan Directory (Test)",
            ]

            choice = unified_prompt(
                "dir_mgmt", "Directory Management", options, allow_back=True
            )

            if not choice or choice == "BACK" or choice == "MAIN_MENU":
                break

            if choice == "Add Directory Path":
                add_directory_path(config_manager)
            elif choice == "Add Current Directory":
                add_current_directory(config_manager)
            elif choice == "Remove Directory":
                remove_directory(config_manager, custom_dirs)
            elif choice == "Toggle Default Cache":
                toggle_default_cache(config_manager, include_default)
            elif choice == "Scan Directory (Test)":
                test_directory_scan(config_manager)

        except KeyboardInterrupt:
            logger.info("Directory management interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in directory management: {e}")
            console.print(f"[red]Error: {e}[/red]")
            input("Press Enter to continue...")


def add_directory_path(config_manager: ConfigManager) -> None:
    """Prompt user to add a custom directory path."""
    console = Console()

    try:
        console.print("\n[bold]Add Directory Path[/bold]")
        console.print(
            "Enter the full path to a directory containing HuggingFace assets:"
        )
        console.print("[dim]Example: /home/user/my-models[/dim]\n")

        # Get path from user
        path_input = input("Path (or 'cancel'): ").strip()

        if path_input.lower() == "cancel":
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Expand user path
        path = Path(path_input).expanduser().resolve()

        # Validate directory
        if not path.exists():
            console.print(f"[red]Error: Directory does not exist: {path}[/red]")
            input("Press Enter to continue...")
            return

        if not path.is_dir():
            console.print(f"[red]Error: Path is not a directory: {path}[/red]")
            input("Press Enter to continue...")
            return

        # Check if it contains HF assets
        if not config_manager.validate_directory(str(path)):
            console.print(
                "[yellow]Warning: Directory doesn't appear to contain HuggingFace assets[/yellow]"
            )
            console.print("Add it anyway? (y/n): ", end="")
            if input().lower() != "y":
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Add directory
        if config_manager.add_directory(str(path)):
            console.print(f"[green]✅ Added directory: {path}[/green]")
        else:
            console.print(f"[yellow]Directory already configured: {path}[/yellow]")

        input("Press Enter to continue...")

    except Exception as e:
        logger.error(f"Error adding directory: {e}")
        console.print(f"[red]Error: {e}[/red]")
        input("Press Enter to continue...")


def add_current_directory(config_manager: ConfigManager) -> None:
    """Add the current working directory to configuration."""
    console = Console()

    try:
        current_dir = os.getcwd()
        console.print(f"\n[bold]Add Current Directory[/bold]")
        console.print(f"Current directory: [cyan]{current_dir}[/cyan]")

        # Check if it contains HF assets
        if not config_manager.validate_directory(current_dir):
            console.print(
                "\n[yellow]Warning: Current directory doesn't appear to contain HuggingFace assets[/yellow]"
            )
            console.print("Add it anyway? (y/n): ", end="")
            if input().lower() != "y":
                console.print("[yellow]Cancelled[/yellow]")
                input("Press Enter to continue...")
                return

        # Add directory
        if config_manager.add_directory(current_dir):
            console.print(f"\n[green]Added current directory to configuration[/green]")
        else:
            console.print(f"\n[yellow]Current directory already configured[/yellow]")

        input("Press Enter to continue...")

    except Exception as e:
        logger.error(f"Error adding current directory: {e}")
        console.print(f"[red]Error: {e}[/red]")
        input("Press Enter to continue...")


def remove_directory(config_manager: ConfigManager, custom_dirs: List[str]) -> None:
    """Remove a directory from configuration."""
    console = Console()

    if not custom_dirs:
        console.print("\n[yellow]No custom directories to remove[/yellow]")
        input("Press Enter to continue...")
        return

    try:
        # Create choices with directory paths
        choices = [f"{i}. {dir_path}" for i, dir_path in enumerate(custom_dirs, 1)]
        choices.append("Cancel")

        choice = unified_prompt(
            "remove_dir", "Select directory to remove", choices, allow_back=False
        )

        if not choice or choice == "Cancel":
            return

        # Extract index from choice
        idx = int(choice.split(".")[0]) - 1
        dir_to_remove = custom_dirs[idx]

        if config_manager.remove_directory(dir_to_remove):
            console.print(f"\n[green]Removed directory: {dir_to_remove}[/green]")
        else:
            console.print(f"\n[red]Failed to remove directory: {dir_to_remove}[/red]")

        input("Press Enter to continue...")

    except Exception as e:
        logger.error(f"Error removing directory: {e}")
        console.print(f"[red]Error: {e}[/red]")
        input("Press Enter to continue...")


def toggle_default_cache(config_manager: ConfigManager, current_state: bool) -> None:
    """Toggle default cache inclusion."""
    console = Console()

    try:
        new_state = config_manager.toggle_default_cache()
        status = "enabled" if new_state else "disabled"
        console.print(f"\n[green]Default HuggingFace cache {status}[/green]")
        input("Press Enter to continue...")

    except Exception as e:
        logger.error(f"Error toggling default cache: {e}")
        console.print(f"[red]Error: {e}[/red]")
        input("Press Enter to continue...")


def test_directory_scan(config_manager: ConfigManager) -> None:
    """Test scanning a directory for HF assets."""
    console = Console()

    try:
        console.print("\n[bold]Test Directory Scan[/bold]")
        console.print("Enter a directory path to test scanning:")

        path_input = input("Path (or 'cancel'): ").strip()

        if path_input.lower() == "cancel":
            return

        path = Path(path_input).expanduser().resolve()

        if not path.exists():
            console.print(f"[red]Directory does not exist: {path}[/red]")
        elif not path.is_dir():
            console.print(f"[red]Path is not a directory: {path}[/red]")
        else:
            # Import here to avoid circular dependency
            from .cache import get_items

            console.print(f"\n[cyan]Scanning {path}...[/cyan]")

            try:
                items = get_items(str(path))

                if items:
                    console.print(f"\n[green]Found {len(items)} assets:[/green]")

                    # Show first 5 items
                    for i, item in enumerate(items[:5]):
                        size_mb = item["size"] / (1024 * 1024)
                        console.print(f"  • {item['name']} ({size_mb:.1f} MB)")

                    if len(items) > 5:
                        console.print(f"  ... and {len(items) - 5} more")
                else:
                    console.print(
                        "[yellow]No HuggingFace assets found in directory[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]Error scanning directory: {e}[/red]")

        input("\nPress Enter to continue...")

    except Exception as e:
        logger.error(f"Error in directory scan test: {e}")
        console.print(f"[red]Error: {e}[/red]")
        input("Press Enter to continue...")


def show_help() -> None:
    """
    Display navigation help and keyboard shortcuts.

    Shows users how to navigate the application interface
    and use available keyboard shortcuts and menu options.
    """
    logger.info("Displaying navigation help")

    try:
        print("\n" + "=" * 50)
        print("NAVIGATION HELP:")
        print("  ↑/↓ arrows: Navigate menu options")
        print("  Enter: Select current option")
        print("  Select '← Back' to go to previous menu")
        print("  Select '→ Config' for settings and options")
        print("  Ctrl+C: Force exit")
        print("=" * 50 + "\n")
        input("Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        logger.info("Help display interrupted by user")
        return


def show_config() -> Optional[str]:
    """
    Display configuration menu with application settings.

    Provides access to sorting options, cache settings, display preferences,
    and help documentation. Returns sort selections for immediate application.

    Returns:
        Sort option string if a sort preference was selected, None otherwise
    """
    logger.info("Displaying configuration menu")

    while True:
        try:
            config_choice = unified_prompt(
                "config",
                "Configuration & Settings",
                [
                    "Sort Assets By Size",
                    "Sort Assets By Date",
                    "Sort Assets By Name",
                    "Manage Cache Directories",
                    "Display Preferences",
                    "Show Help",
                ],
                allow_back=True,
            )

            if not config_choice or config_choice == "BACK":
                logger.info("User exited configuration menu")
                break

            logger.info(f"User selected config option: {config_choice}")

            if config_choice.startswith("Sort Assets"):
                # Return the sort choice to be used by the calling function
                return config_choice
            elif config_choice == "Manage Cache Directories":
                manage_directories()
            elif config_choice == "Display Preferences":
                print("\n[Future Feature] Display preferences")
                print("This will allow you to customize how assets are displayed.")
                try:
                    input("Press Enter to continue...")
                except (KeyboardInterrupt, EOFError):
                    break
            elif config_choice == "Show Help":
                show_help()

        except KeyboardInterrupt:
            logger.info("Configuration menu interrupted by user")
            break

    return None


def unified_prompt(
    name: str, message: str, choices: List[str], allow_back: bool = True
) -> Optional[str]:
    """
    Unified prompt with consistent navigation across all menus.

    Provides standardized menu interface with navigation options,
    configuration access, and consistent user experience throughout the application.

    Args:
        name: Unique identifier for the prompt
        message: Question or menu title to display
        choices: List of menu options to present
        allow_back: Whether to show the Back option

    Returns:
        Selected choice string, or special navigation constants:
        - 'BACK': User selected back navigation
        - 'MAIN_MENU': User wants to return to main menu
        - Sort option strings from configuration
        - None: User cancelled or interrupted
    """
    if not isinstance(choices, list):
        raise ValueError("Choices must be a list")

    logger.debug(f"Creating unified prompt '{name}' with {len(choices)} choices")

    # Create enhanced choices with navigation
    enhanced_choices = list(choices)

    # Remove existing navigation options to avoid duplicates
    enhanced_choices = [
        c
        for c in enhanced_choices
        if c not in ["Back", "Help", "Quit", "← Back", "→ Config", "Main Menu", "Exit"]
    ]

    # Add separator and navigation options
    enhanced_choices.append("─────")
    if allow_back:
        enhanced_choices.append("← Back")
    enhanced_choices.append("→ Config")
    enhanced_choices.append("Main Menu")
    enhanced_choices.append("Exit")

    # Create custom theme for clean appearance
    custom_theme = inquirer.themes.GreenPassion()
    # Try to customize the prompt appearance
    if hasattr(custom_theme.Question, "mark"):
        custom_theme.Question.mark = "🎯"

    console = Console()

    while True:
        try:
            # Display compact menu title in a panel
            console.print(
                Panel(
                    f"[bold white]{message}[/bold white]",
                    border_style="bright_blue",
                    padding=(0, 1),
                    expand=False,
                )
            )

            question = inquirer.List(
                name,
                message="Select an option",
                choices=enhanced_choices,
                carousel=True,
            )

            answers = inquirer.prompt([question], theme=custom_theme)
            if not answers:
                logger.info("User cancelled prompt")
                return None

            result: str = answers[name]
            logger.debug(f"User selected: {result}")

            # Handle special navigation choices
            if result == "← Back":
                return "BACK"
            elif result == "→ Config":
                config_result = show_config()
                if config_result:  # If a sort option was selected
                    return config_result
                continue  # Stay in current menu if config was just browsed
            elif result == "Main Menu":
                return "MAIN_MENU"
            elif result == "Exit":
                logger.info("User selected exit")
                sys.exit(0)
            elif result == "─────":
                continue  # Ignore separator selection
            else:
                return result

        except KeyboardInterrupt:
            logger.info("Prompt interrupted by user")
            sys.exit(0)
        except Exception as e:
            # Handle ioctl errors gracefully (common in non-terminal environments)
            if "Inappropriate ioctl for device" in str(e):
                logger.warning("Running in non-interactive environment")
                return None
            logger.error(f"Error in unified prompt: {e}")
            return None
