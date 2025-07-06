#!/usr/bin/env python3
"""
User interface module for HF-MODEL-TOOL.

Provides rich terminal-based user interfaces for asset management,
including listing, deletion, deduplication, and detailed asset viewing.
"""
import os
import json
import shutil
import logging
from typing import List, Dict, Any, Optional

import inquirer
import html2text
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .utils import group_and_identify_duplicates
from .navigation import unified_prompt

logger = logging.getLogger(__name__)

# Legacy constant for backward compatibility
BACK_CHOICE = "Back"


def print_items(items: List[Dict[str, Any]], sort_by: str = "size") -> None:
    """
    Display a formatted table of assets grouped by category and publisher.

    Args:
        items: List of asset dictionaries from cache scanning
        sort_by: Sort criteria - 'size', 'date', or 'name'

    Raises:
        ValueError: If sort_by is not a valid option
    """
    if sort_by not in ["size", "date", "name"]:
        raise ValueError(
            f"Invalid sort_by option: {sort_by}. Must be 'size', 'date', or 'name'"
        )

    logger.info(f"Displaying {len(items)} assets sorted by {sort_by}")

    console = Console()

    try:
        total_size = sum(item.get("size", 0) for item in items)
        console.print(
            Panel(
                f"[bold cyan]Grand Total All Assets: {total_size / 1e9:.2f} GB[/bold cyan]",
                expand=False,
            )
        )

        grouped, _ = group_and_identify_duplicates(items)

        sorted_categories = sorted(
            grouped.items(),
            key=lambda x: sum(
                item["size"] for pub_items in x[1].values() for item in pub_items
            ),
            reverse=(sort_by == "size"),
        )

        for category, publishers in sorted_categories:
            if not publishers:
                continue

            category_size = sum(
                item["size"] for pub_items in publishers.values() for item in pub_items
            )

            table = Table(
                title=f"[bold green]{category.upper()} (Total: {category_size / 1e9:.2f} GB)[/bold green]"
            )
            table.add_column("Publisher/Name", style="cyan", no_wrap=True)
            table.add_column("Size (GB)", style="magenta", justify="right")
            table.add_column("Modified Date", style="yellow", justify="right")
            table.add_column("Notes", style="red")

            sorted_publishers = sorted(
                publishers.items(),
                key=lambda x: sum(item["size"] for item in x[1]),
                reverse=(sort_by == "size"),
            )

            for publisher, item_list in sorted_publishers:
                publisher_size = sum(item["size"] for item in item_list)
                table.add_row(
                    f"[bold blue]Publisher: {publisher} (Total: {publisher_size / 1e9:.2f} GB)[/bold blue]"
                )

                if sort_by == "size":
                    sorted_list = sorted(
                        item_list, key=lambda x: x["size"], reverse=True
                    )
                elif sort_by == "date":
                    sorted_list = sorted(
                        item_list, key=lambda x: x["date"], reverse=True
                    )
                else:  # name
                    sorted_list = sorted(item_list, key=lambda x: x["display_name"])

                for item in sorted_list:
                    duplicate_marker = "(duplicate)" if item["is_duplicate"] else ""
                    table.add_row(
                        f"  {item['display_name']}",
                        f"{item['size'] / 1e9:.2f}",
                        item["date"].strftime("%Y-%m-%d %H:%M:%S"),
                        duplicate_marker,
                    )

            console.print(table)

    except Exception as e:
        logger.error(f"Error displaying assets: {e}")
        console.print(f"[red]Error displaying assets: {e}[/red]")


def delete_assets_workflow(items: List[Dict[str, Any]]) -> Optional[str]:
    if not items:
        print("No assets to delete.")
        return None

    grouped, _ = group_and_identify_duplicates(items)

    while True:  # Main delete loop
        category_choices = [cat.capitalize() for cat in grouped.keys() if grouped[cat]]
        category = unified_prompt(
            "category",
            "Select Category to Delete From",
            category_choices,
            allow_back=True,
        )
        if not category or category == "BACK":
            break
        elif category == "MAIN_MENU":
            return "MAIN_MENU"
        selected_category = category.lower()

        while True:  # Publisher loop
            publisher_choices = list(grouped[selected_category].keys())
            publisher = unified_prompt(
                "publisher",
                f"Select Publisher from {category}",
                publisher_choices,
                allow_back=True,
            )
            if not publisher or publisher == "BACK":
                break
            elif publisher == "MAIN_MENU":
                return "MAIN_MENU"
            selected_publisher = publisher

            while True:  # Item loop
                items_to_delete_choices = grouped[selected_category][selected_publisher]
                choices = [
                    f"{item['display_name']} ({item['size']/1e9:.2f} GB)"
                    for item in items_to_delete_choices
                ]
                questions = [
                    inquirer.Checkbox(
                        "selected_items",
                        message="Select assets to delete (space to select, enter to confirm)",
                        choices=choices,
                    )
                ]
                answers = inquirer.prompt(questions)
                if not answers:
                    break  # User pressed Ctrl+C

                if not answers["selected_items"]:
                    action_choice = unified_prompt(
                        "action",
                        "Nothing selected.",
                        ["Go back and select assets", "Return to publisher menu"],
                        allow_back=False,
                    )
                    if not action_choice:
                        break  # Exit item loop if no choice made
                    elif action_choice == "Go back and select assets":
                        continue  # Restart item loop
                    elif action_choice == "Return to publisher menu":
                        break  # Exit item loop, back to publisher
                    elif action_choice == "MAIN_MENU":
                        return "MAIN_MENU"  # Exit the entire delete workflow back to main menu

                confirm = inquirer.confirm(
                    f"Are you sure you want to delete {len(answers['selected_items'])} assets?",
                    default=False,
                )
                if confirm:
                    for choice_str in answers["selected_items"]:
                        item_name_to_find = choice_str.split(" ")[0]
                        for item in items_to_delete_choices:
                            if item["display_name"] == item_name_to_find:
                                shutil.rmtree(item["path"])
                                print(f"Removed: {item['name']}")
                                break
                else:
                    print("Deletion cancelled.")
                break  # Exit item loop after action

    return None


def deduplicate_assets_workflow(items: List[Dict[str, Any]]) -> Optional[str]:
    _, duplicate_sets = group_and_identify_duplicates(items)
    if not duplicate_sets:
        print("No duplicates found.")
        return None

    print(f"Found {len(duplicate_sets)} set(s) of duplicates.")
    for dup_set in duplicate_sets:
        dup_items = [item for item in items if item["name"] in dup_set]
        dup_items.sort(key=lambda x: x["date"], reverse=True)

        choices = [
            f"{i['name']} ({i['date'].strftime('%Y-%m-%d')}, {i['size']/1e9:.2f} GB)"
            for i in dup_items
        ]
        keep_choice = unified_prompt(
            "item_to_keep",
            f"Select version of '{dup_items[0]['display_name']}' to KEEP (newest is default)",
            choices,
            allow_back=True,
        )
        if not keep_choice or keep_choice == "BACK":
            continue
        elif keep_choice == "MAIN_MENU":
            return "MAIN_MENU"

        item_to_keep_name = keep_choice.split(" ")[0]
        items_to_delete = [
            item for item in dup_items if item["name"] != item_to_keep_name
        ]

        print("The following assets will be deleted:")
        for item in items_to_delete:
            print(f"- {item['name']}")

        confirm = inquirer.confirm(
            f"Are you sure you want to delete {len(items_to_delete)} duplicate(s)?",
            default=False,
        )
        if confirm:
            for item in items_to_delete:
                shutil.rmtree(item["path"])
                print(f"Removed duplicate: {item['name']}")
        else:
            print("Deduplication for this set cancelled.")
    print("Deduplication complete.")
    return None


def view_asset_details_workflow(items: List[Dict[str, Any]]) -> Optional[str]:
    if not items:
        print("No assets to view.")
        return None

    grouped, _ = group_and_identify_duplicates(items)

    while True:  # Category loop
        category_choices = [cat.capitalize() for cat in grouped.keys() if grouped[cat]]
        if not category_choices:
            print("No assets to view.")
            return None

        category = unified_prompt(
            "category", "Select Category to View", category_choices, allow_back=True
        )
        if not category or category == "BACK":
            break
        elif category == "MAIN_MENU":
            return "MAIN_MENU"
        selected_category_name = category.lower()

        assets_in_category = grouped.get(selected_category_name, {})

        while True:  # Publisher loop
            publisher_choices = list(assets_in_category.keys())
            publisher = unified_prompt(
                "publisher",
                f"Select Publisher from {category}",
                publisher_choices,
                allow_back=True,
            )
            if not publisher or publisher == "BACK":
                break
            elif publisher == "MAIN_MENU":
                return "MAIN_MENU"
            selected_publisher = publisher

            while True:  # Item loop
                asset_choices = assets_in_category[selected_publisher]
                choices = [
                    f"{item['display_name']} ({item['size']/1e9:.2f} GB)"
                    for item in asset_choices
                ]
                selected_asset_str = unified_prompt(
                    "selected_asset",
                    f"Select Asset from {selected_publisher}",
                    choices,
                    allow_back=True,
                )
                if not selected_asset_str or selected_asset_str == "BACK":
                    break
                elif selected_asset_str == "MAIN_MENU":
                    return "MAIN_MENU"

                selected_asset_display_name = selected_asset_str.split(" ")[0]
                selected_asset = next(
                    (
                        item
                        for item in asset_choices
                        if item["display_name"] == selected_asset_display_name
                    ),
                    None,
                )

                if selected_asset:
                    console = Console()
                    asset_type = selected_asset["type"]

                    if asset_type == "model":
                        file_to_find = "config.json"
                    else:  # dataset
                        file_to_find = "README.md"

                    file_path = None
                    for root, _, files in os.walk(selected_asset["path"]):
                        if file_to_find in files:
                            file_path = os.path.join(root, file_to_find)
                            break

                    if file_path and os.path.exists(file_path):
                        if asset_type == "model":
                            with open(file_path, "r") as f:
                                config_data = json.load(f)

                            console.print(
                                Panel(
                                    f"[bold cyan]Configuration for {selected_asset['name']}[/bold cyan]\n[yellow]Path:[/] {file_path}",
                                    expand=False,
                                )
                            )
                            quant_config = config_data.pop("quantization_config", None)

                            # Main config table
                            main_config_table = Table(
                                title="[bold green]Main Configuration[/bold green]",
                                show_header=True,
                                header_style="bold blue",
                            )
                            main_config_table.add_column(
                                "Parameter", style="cyan", no_wrap=True
                            )
                            main_config_table.add_column("Value", style="magenta")

                            for key, value in sorted(config_data.items()):
                                if isinstance(value, list):
                                    main_config_table.add_row(
                                        key, "\n".join(map(str, value))
                                    )
                                elif isinstance(value, dict):
                                    main_config_table.add_row(
                                        key, json.dumps(value, indent=2)
                                    )
                                else:
                                    main_config_table.add_row(key, str(value))
                            console.print(main_config_table)

                            # Quantization config table
                            if quant_config:
                                quant_table = Table(
                                    title="[bold green]Quantization Configuration[/bold green]",
                                    show_header=True,
                                    header_style="bold blue",
                                )
                                quant_table.add_column(
                                    "Parameter", style="cyan", no_wrap=True
                                )
                                quant_table.add_column("Value", style="magenta")
                                for key, value in sorted(quant_config.items()):
                                    if isinstance(value, list):
                                        quant_table.add_row(
                                            key, "\n".join(map(str, value))
                                        )
                                    elif isinstance(value, dict):
                                        quant_table.add_row(
                                            key, json.dumps(value, indent=2)
                                        )
                                    else:
                                        quant_table.add_row(key, str(value))
                                console.print(quant_table)

                        else:  # dataset
                            with open(file_path, "r", encoding="utf-8") as f:
                                readme_content = f.read()

                            # Check if content is already markdown or needs conversion
                            if (
                                readme_content.strip().startswith("<!DOCTYPE html>")
                                or "<html" in readme_content.lower()
                            ):
                                # Convert HTML to markdown for better display
                                h = html2text.HTML2Text()
                                h.ignore_links = False
                                h.ignore_images = True
                                h.body_width = 0  # Don't wrap lines
                                h.unicode_snob = True
                                markdown_content = h.handle(readme_content)
                            else:
                                # Already markdown or plain text
                                markdown_content = readme_content

                            # Use Rich's markdown renderer within a panel
                            try:
                                md = Markdown(markdown_content)
                                console.print(
                                    Panel(
                                        md,
                                        title=f"[bold cyan]Details for {selected_asset['name']}[/bold cyan]",
                                        subtitle=f"[yellow]Path:[/] {file_path}",
                                        expand=False,
                                    )
                                )
                            except Exception:
                                # Fallback to plain text if markdown parsing fails
                                console.print(
                                    Panel(
                                        markdown_content,
                                        title=f"[bold cyan]Details for {selected_asset['name']} (Plain Text)[/bold cyan]",
                                        subtitle=f"[yellow]Path:[/] {file_path}",
                                        expand=False,
                                    )
                                )
                    else:
                        console.print(
                            Panel(
                                f"[bold red]No {file_to_find} found for {selected_asset['name']}[/bold red]",
                                expand=False,
                            )
                        )

                    input("\nPress Enter to continue...")
                break

    return None
