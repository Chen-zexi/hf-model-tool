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
from pathlib import Path
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
            
            # Create category-specific titles
            category_titles = {
                "models": "HUGGINGFACE MODELS",
                "datasets": "HUGGINGFACE DATASETS", 
                "lora_adapters": "LORA ADAPTERS",
                "custom_models": "CUSTOM MODELS",
                "unknown_models": "UNKNOWN MODELS",
                "unknown": "UNKNOWN ASSETS"
            }
            
            display_title = category_titles.get(category, category.upper())

            table = Table(
                title=f"[bold green]{display_title} (Total: {category_size / 1e9:.2f} GB)[/bold green]"
            )
            table.add_column("Publisher/Name", style="cyan", no_wrap=True)
            table.add_column("Size (GB)", style="magenta", justify="right")
            table.add_column("Modified Date", style="yellow", justify="right")
            table.add_column("Type/Notes", style="red")

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
                    # Create type/notes column with asset type and duplicate info
                    type_info = item.get("subtype", "unknown")
                    duplicate_marker = " (duplicate)" if item["is_duplicate"] else ""
                    notes = f"{type_info}{duplicate_marker}"
                    
                    # Add metadata info for specific asset types
                    if item.get("type") == "lora_adapter":
                        metadata = item.get("metadata", {})
                        rank = metadata.get("lora_rank", "unknown")
                        notes = f"LoRA (rank={rank}){duplicate_marker}"
                    elif item.get("type") == "custom_model":
                        subtype = item.get("subtype", "custom")
                        notes = f"Custom ({subtype}){duplicate_marker}"
                    
                    table.add_row(
                        f"  {item['display_name']}",
                        f"{item['size'] / 1e9:.2f}",
                        item["date"].strftime("%Y-%m-%d %H:%M:%S"),
                        notes,
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
                        # Extract display name by removing the size info in parentheses at the end
                        # Format: "display_name (size GB)" -> "display_name"
                        if " (" in choice_str and choice_str.endswith(" GB)"):
                            # Remove the last parentheses group (size info)
                            item_name_to_find = choice_str.rsplit(" (", 1)[0]
                        else:
                            # Fallback: use the full string for exact matching
                            item_name_to_find = choice_str
                            
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

        # Extract item name by removing the date and size info in parentheses at the end  
        # Format: "name (date, size GB)" -> "name"
        item_to_keep_name = keep_choice.split(" (")[0]
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

                # Extract the display name by removing the size info in parentheses at the end
                # Format: "lora_name (timestamp) (size GB)" -> "lora_name (timestamp)"
                # Format: "model_name (size GB)" -> "model_name"
                if " (" in selected_asset_str and selected_asset_str.endswith(" GB)"):
                    # Remove the last parentheses group (size info)
                    selected_asset_display_name = selected_asset_str.rsplit(" (", 1)[0]
                else:
                    # Fallback: use the full string minus size info for exact matching
                    selected_asset_display_name = selected_asset_str
                
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
                    
                    # Display asset information based on type
                    if asset_type == "lora_adapter":
                        _display_lora_details(console, selected_asset)
                    elif asset_type == "custom_model":
                        _display_custom_model_details(console, selected_asset)
                    elif asset_type == "model":
                        _display_standard_model_details(console, selected_asset)
                    elif asset_type == "dataset":
                        _display_dataset_details(console, selected_asset)
                    else:
                        _display_generic_details(console, selected_asset)

                    input("\nPress Enter to continue...")
                break

    return None


def _display_lora_details(console: Console, asset: Dict[str, Any]) -> None:
    """Display detailed information for LoRA adapter assets."""
    # Use the specific LoRA path if available, otherwise use the main path
    lora_path = asset.get("lora_path", asset["path"])
    
    console.print(
        Panel(
            f"[bold cyan]LoRA Adapter: {asset['display_name']}[/bold cyan]\n"
            f"[yellow]Path:[/] {lora_path}\n"
            f"[yellow]Size:[/] {asset['size'] / 1e9:.2f} GB",
            expand=False,
        )
    )
    
    # Show full adapter configuration from JSON file
    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r") as f:
                config_data = json.load(f)
            
            console.print(
                Panel(
                    f"[bold cyan]Adapter Configuration[/bold cyan]\n"
                    f"[yellow]File:[/] {adapter_config_path}",
                    expand=False,
                )
            )
            
            # Create a comprehensive table showing all configuration
            config_table = Table(
                title="[bold green]LoRA Adapter Configuration[/bold green]",
                show_header=True,
                header_style="bold blue",
                width=120,
            )
            config_table.add_column("Parameter", style="cyan", no_wrap=True, width=25)
            config_table.add_column("Value", style="magenta", width=95)
            
            # Sort and display all configuration parameters
            for key, value in sorted(config_data.items()):
                if isinstance(value, list):
                    # For lists, show each item on a new line
                    if len(value) > 0:
                        formatted_value = "\n".join(f"• {str(item)}" for item in value)
                    else:
                        formatted_value = "(empty list)"
                elif isinstance(value, dict):
                    # For dictionaries, show as formatted JSON
                    if value:
                        formatted_value = json.dumps(value, indent=2)
                    else:
                        formatted_value = "(empty object)"
                elif isinstance(value, bool):
                    # Format boolean values clearly
                    formatted_value = "✓ True" if value else "✗ False"
                elif value is None:
                    formatted_value = "(null)"
                else:
                    formatted_value = str(value)
                
                config_table.add_row(key, formatted_value)
            
            console.print(config_table)
            
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing adapter config JSON: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Error reading adapter config: {e}[/red]")
    else:
        console.print(
            Panel(
                f"[bold red]No adapter_config.json found in {asset['path']}[/bold red]",
                expand=False,
            )
        )
    
    # Also show any other relevant files
    asset_path = Path(lora_path)
    relevant_files = []
    for file_pattern in ["*.safetensors", "*.bin", "*.json", "README.md"]:
        relevant_files.extend(asset_path.glob(file_pattern))
    
    if relevant_files:
        files_table = Table(
            title="[bold green]Files in LoRA Adapter[/bold green]",
            show_header=True,
            header_style="bold blue",
        )
        files_table.add_column("File", style="cyan")
        files_table.add_column("Size", style="magenta", justify="right")
        
        for file_path in sorted(relevant_files):
            try:
                size = file_path.stat().st_size
                if size > 1e6:  # > 1MB
                    size_str = f"{size / 1e6:.1f} MB"
                elif size > 1e3:  # > 1KB
                    size_str = f"{size / 1e3:.1f} KB"
                else:
                    size_str = f"{size} B"
                files_table.add_row(file_path.name, size_str)
            except OSError:
                files_table.add_row(file_path.name, "N/A")
        
        console.print(files_table)


def _display_custom_model_details(console: Console, asset: Dict[str, Any]) -> None:
    """Display detailed information for custom model assets."""
    metadata = asset.get("metadata", {})
    
    console.print(
        Panel(
            f"[bold cyan]Custom Model Details: {asset['display_name']}[/bold cyan]\n"
            f"[yellow]Path:[/] {asset['path']}\n"
            f"[yellow]Size:[/] {asset['size'] / 1e9:.2f} GB\n"
            f"[yellow]Type:[/] {asset.get('subtype', 'unknown')}",
            expand=False,
        )
    )
    
    # Model Configuration Table
    if metadata:
        model_table = Table(
            title="[bold green]Model Configuration[/bold green]",
            show_header=True,
            header_style="bold blue",
        )
        model_table.add_column("Parameter", style="cyan", no_wrap=True)
        model_table.add_column("Value", style="magenta")
        
        model_params = [
            ("Model Type", metadata.get("model_type", "unknown")),
            ("Architectures", ", ".join(metadata.get("architectures", []))),
            ("Torch Dtype", metadata.get("torch_dtype", "unknown")),
            ("Vocab Size", metadata.get("vocab_size", "unknown")),
        ]
        
        # Add fine-tuning framework info if available
        if metadata.get("fine_tuning_framework"):
            model_params.append(("Fine-tuning Framework", metadata.get("fine_tuning_framework")))
            if metadata.get("unsloth_version"):
                model_params.append(("Unsloth Version", metadata.get("unsloth_version")))
        
        for param, value in model_params:
            model_table.add_row(param, str(value))
        
        console.print(model_table)
    
    # Show config file if available
    _display_config_file(console, asset)


def _display_standard_model_details(console: Console, asset: Dict[str, Any]) -> None:
    """Display detailed information for standard HuggingFace model assets."""
    console.print(
        Panel(
            f"[bold cyan]HuggingFace Model: {asset['display_name']}[/bold cyan]\n"
            f"[yellow]Path:[/] {asset['path']}\n"
            f"[yellow]Size:[/] {asset['size'] / 1e9:.2f} GB",
            expand=False,
        )
    )
    
    _display_config_file(console, asset)


def _display_dataset_details(console: Console, asset: Dict[str, Any]) -> None:
    """Display detailed information for dataset assets."""
    console.print(
        Panel(
            f"[bold cyan]HuggingFace Dataset: {asset['display_name']}[/bold cyan]\n"
            f"[yellow]Path:[/] {asset['path']}\n"
            f"[yellow]Size:[/] {asset['size'] / 1e9:.2f} GB",
            expand=False,
        )
    )
    
    # Look for README.md
    readme_path = None
    for root, _, files in os.walk(asset["path"]):
        if "README.md" in files:
            readme_path = os.path.join(root, "README.md")
            break
    
    if readme_path and os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
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
                        title=f"[bold cyan]Dataset Information[/bold cyan]",
                        subtitle=f"[yellow]Path:[/] {readme_path}",
                        expand=False,
                    )
                )
            except Exception:
                # Fallback to plain text if markdown parsing fails
                console.print(
                    Panel(
                        markdown_content,
                        title=f"[bold cyan]Dataset Information (Plain Text)[/bold cyan]",
                        subtitle=f"[yellow]Path:[/] {readme_path}",
                        expand=False,
                    )
                )
        except Exception as e:
            console.print(f"[red]Error reading README: {e}[/red]")
    else:
        console.print(
            Panel(
                f"[bold red]No README.md found for {asset['display_name']}[/bold red]",
                expand=False,
            )
        )


def _display_generic_details(console: Console, asset: Dict[str, Any]) -> None:
    """Display generic information for unknown asset types."""
    console.print(
        Panel(
            f"[bold cyan]Asset Details: {asset['display_name']}[/bold cyan]\n"
            f"[yellow]Path:[/] {asset['path']}\n"
            f"[yellow]Size:[/] {asset['size'] / 1e9:.2f} GB\n"
            f"[yellow]Type:[/] {asset.get('type', 'unknown')}",
            expand=False,
        )
    )
    
    # Show available files
    files = asset.get("files", [])
    if files:
        files_table = Table(
            title="[bold green]Files Found[/bold green]",
            show_header=True,
            header_style="bold blue",
        )
        files_table.add_column("File", style="cyan")
        
        for file in files[:20]:  # Show first 20 files
            files_table.add_row(file)
        
        if len(files) > 20:
            files_table.add_row(f"... and {len(files) - 20} more files")
        
        console.print(files_table)


def _display_config_file(console: Console, asset: Dict[str, Any]) -> None:
    """Display config.json file contents if available."""
    config_path = None
    for root, _, files in os.walk(asset["path"]):
        if "config.json" in files:
            config_path = os.path.join(root, "config.json")
            break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            console.print(
                Panel(
                    f"[bold cyan]Configuration File[/bold cyan]\n[yellow]Path:[/] {config_path}",
                    expand=False,
                )
            )
            
            # Separate quantization config if present
            quant_config = config_data.pop("quantization_config", None)
            
            # Main config table
            main_config_table = Table(
                title="[bold green]Main Configuration[/bold green]",
                show_header=True,
                header_style="bold blue",
            )
            main_config_table.add_column("Parameter", style="cyan", no_wrap=True)
            main_config_table.add_column("Value", style="magenta")
            
            for key, value in sorted(config_data.items()):
                if isinstance(value, list):
                    main_config_table.add_row(key, "\n".join(map(str, value)))
                elif isinstance(value, dict):
                    main_config_table.add_row(key, json.dumps(value, indent=2))
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
                quant_table.add_column("Parameter", style="cyan", no_wrap=True)
                quant_table.add_column("Value", style="magenta")
                
                for key, value in sorted(quant_config.items()):
                    if isinstance(value, list):
                        quant_table.add_row(key, "\n".join(map(str, value)))
                    elif isinstance(value, dict):
                        quant_table.add_row(key, json.dumps(value, indent=2))
                    else:
                        quant_table.add_row(key, str(value))
                
                console.print(quant_table)
        
        except Exception as e:
            console.print(f"[red]Error reading config file: {e}[/red]")
    else:
        console.print(
            Panel(
                f"[bold red]No config.json found for {asset['display_name']}[/bold red]",
                expand=False,
            )
        )
