
#!/usr/bin/env python3
import os
import shutil
import inquirer
from datetime import datetime
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

BACK_CHOICE = "[<- Back]"

def get_items(cache_dir):
    items = []
    for item_dir in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item_dir)
        if os.path.isdir(item_path):
            blobs_path = os.path.join(item_path, "blobs")
            size = 0
            if os.path.isdir(blobs_path):
                try:
                    size = sum(os.path.getsize(os.path.join(blobs_path, filename)) for filename in os.listdir(blobs_path))
                except FileNotFoundError:
                    pass # Ignore issues with concurrently deleted files
            
            if size > 0:
                items.append({
                    "name": item_dir,
                    "size": size,
                    "date": datetime.fromtimestamp(os.path.getmtime(item_path)),
                    "type": "dataset" if item_dir.lower().startswith("datasets--") else "model",
                    "path": item_path
                })
    return items

def group_and_identify_duplicates(items):
    grouped_for_dupes = defaultdict(list)
    for item in items:
        parts = item["name"].lower().split("--")
        if len(parts) > 2:
            key = (parts[0], parts[1], '--'.join(parts[2:]))
            grouped_for_dupes[key].append(item["name"])

    duplicate_sets = {frozenset(v) for v in grouped_for_dupes.values() if len(v) > 1}
    is_duplicate = {name for dup_set in duplicate_sets for name in dup_set}

    grouped_for_display = {"models": defaultdict(list), "datasets": defaultdict(list)}
    for item in items:
        parts = item["name"].split("--")
        if len(parts) > 1:
            publisher = parts[1]
            item['display_name'] = '--'.join(parts[2:]) if len(parts) > 2 else item['name']
            item['is_duplicate'] = item['name'] in is_duplicate
            category = item["type"] + "s"
            grouped_for_display[category][publisher].append(item)
            
    return grouped_for_display, duplicate_sets

def print_items(items, sort_by='size'):
    console = Console()
    total_size = sum(item['size'] for item in items)
    console.print(Panel(f"[bold cyan]Grand Total All Assets: {total_size / 1e9:.2f} GB[/bold cyan]", expand=False))

    grouped, _ = group_and_identify_duplicates(items)
    
    sorted_categories = sorted(grouped.items(), key=lambda x: sum(item['size'] for pub_items in x[1].values() for item in pub_items), reverse=(sort_by == 'size'))

    for category, publishers in sorted_categories:
        if not publishers:
            continue
        
        category_size = sum(item['size'] for pub_items in publishers.values() for item in pub_items)
        
        table = Table(title=f"[bold green]{category.upper()} (Total: {category_size / 1e9:.2f} GB)[/bold green]")
        table.add_column("Publisher/Name", style="cyan", no_wrap=True)
        table.add_column("Size (GB)", style="magenta", justify="right")
        table.add_column("Modified Date", style="yellow", justify="right")
        table.add_column("Notes", style="red")

        sorted_publishers = sorted(publishers.items(), key=lambda x: sum(item['size'] for item in x[1]), reverse=(sort_by == 'size'))

        for publisher, item_list in sorted_publishers:
            publisher_size = sum(item['size'] for item in item_list)
            table.add_row(f"[bold blue]Publisher: {publisher} (Total: {publisher_size / 1e9:.2f} GB)[/bold blue]")

            if sort_by == 'size':
                sorted_list = sorted(item_list, key=lambda x: x['size'], reverse=True)
            elif sort_by == 'date':
                sorted_list = sorted(item_list, key=lambda x: x['date'], reverse=True)
            else: # name
                sorted_list = sorted(item_list, key=lambda x: x['display_name'])

            for item in sorted_list:
                duplicate_marker = "(duplicate)" if item['is_duplicate'] else ""
                table.add_row(f"  {item['display_name']}", f"{item['size'] / 1e9:.2f}", item['date'].strftime('%Y-%m-%d %H:%M:%S'), duplicate_marker)
        
        console.print(table)

def delete_items_workflow(items, cache_dir):
    if not items:
        print("No items to delete.")
        return

    grouped, _ = group_and_identify_duplicates(items)

    while True: # Main delete loop
        category_choices = [cat.capitalize() for cat in grouped.keys() if grouped[cat]] + [BACK_CHOICE]
        questions = [inquirer.List('category', message="Which category to delete from?", choices=category_choices, carousel=True)]
        answers = inquirer.prompt(questions)
        if not answers or answers['category'] == BACK_CHOICE: break
        selected_category = answers['category'].lower()

        while True: # Publisher loop
            publisher_choices = list(grouped[selected_category].keys()) + [BACK_CHOICE]
            questions = [inquirer.List('publisher', message="Which publisher?", choices=publisher_choices, carousel=True)]
            answers = inquirer.prompt(questions)
            if not answers or answers['publisher'] == BACK_CHOICE: break
            selected_publisher = answers['publisher']

            while True: # Item loop
                items_to_delete_choices = grouped[selected_category][selected_publisher]
                choices = [f"{item['display_name']} ({item['size']/1e9:.2f} GB)" for item in items_to_delete_choices]
                questions = [inquirer.Checkbox('selected_items', message="Select items to delete (space to select, enter to confirm)", choices=choices)]
                answers = inquirer.prompt(questions)
                if not answers: break # User pressed Ctrl+C

                if not answers['selected_items']:
                    q = [inquirer.List('action', message="Nothing selected.", choices=["Go back and select items", "Return to publisher menu"], carousel=True)]
                    a = inquirer.prompt(q)
                    if not a or a['action'] == "Return to publisher menu":
                        break # Exit item loop, back to publisher
                    else:
                        continue # Restart item loop

                confirm = inquirer.confirm(f"Are you sure you want to delete {len(answers['selected_items'])} items?", default=False)
                if confirm:
                    for choice_str in answers['selected_items']:
                        item_name_to_find = choice_str.split(' ')[0]
                        for item in items_to_delete_choices:
                            if item['display_name'] == item_name_to_find:
                                shutil.rmtree(item['path'])
                                print(f"Removed: {item['name']}")
                                break
                else:
                    print("Deletion cancelled.")
                break # Exit item loop after action

def main():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    while True:
        questions = [
            inquirer.List('action',
                          message="What do you want to do?",
                          choices=[
                              'List by size (default)',
                              'List by date',
                              'List by name',
                              'Delete items...',
                              'Deduplicate items',
                              'Exit'
                          ],
            ),
        ]
        answers = inquirer.prompt(questions)
        if not answers: break
        action = answers['action']

        items = get_items(cache_dir)

        if action.startswith('List'):
            sort_by = 'size'
            if 'date' in action: sort_by = 'date'
            if 'name' in action: sort_by = 'name'
            print_items(items, sort_by=sort_by)

        elif action == 'Delete items...':
            delete_items_workflow(items, cache_dir)

        elif action == 'Deduplicate items':
            _, duplicate_sets = group_and_identify_duplicates(items)
            if not duplicate_sets: print("No duplicates found."); continue

            print(f"Found {len(duplicate_sets)} set(s) of duplicates.")
            for dup_set in duplicate_sets:
                dup_items = [item for item in items if item['name'] in dup_set]
                dup_items.sort(key=lambda x: x['date'], reverse=True)
                
                choices = [f"{i['name']} ({i['date'].strftime('%Y-%m-%d')}, {i['size']/1e9:.2f} GB)" for i in dup_items] + [BACK_CHOICE]
                questions = [inquirer.List('item_to_keep', message=f"Select version of '{dup_items[0]['display_name']}' to KEEP (newest is default)", choices=choices, carousel=True)]
                answers = inquirer.prompt(questions)
                if not answers or answers['item_to_keep'] == BACK_CHOICE: continue

                item_to_keep_name = answers['item_to_keep'].split(' ')[0]
                items_to_delete = [item for item in dup_items if item['name'] != item_to_keep_name]
                
                print("The following items will be deleted:")
                for item in items_to_delete:
                    print(f"- {item['name']}")
                
                confirm = inquirer.confirm(f"Are you sure you want to delete {len(items_to_delete)} duplicate(s)?", default=False)
                if confirm:
                    for item in items_to_delete:
                        shutil.rmtree(item['path'])
                        print(f"Removed duplicate: {item['name']}")
                else:
                    print("Deduplication for this set cancelled.")
            print("Deduplication complete.")

        elif action == 'Exit':
            break

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
