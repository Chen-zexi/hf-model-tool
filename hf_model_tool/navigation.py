#!/usr/bin/env python3
"""
Unified navigation system for HF-MODEL-TOOL.

Provides consistent menu navigation, configuration management,
and help system across all application workflows.
"""
import sys
import logging
from typing import Optional, List

import inquirer
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


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
                    "Set Default Cache Directory",
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
            elif config_choice == "Set Default Cache Directory":
                print("\n[Future Feature] Cache directory configuration")
                print(
                    "This will allow you to change the default HuggingFace cache location."
                )
                try:
                    input("Press Enter to continue...")
                except (KeyboardInterrupt, EOFError):
                    break
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
