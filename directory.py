import os

def print_directory_structure(root_dir, prefix=""):
    """
    Print the directory structure of root_dir with proper lines and indentations,
    ignoring .txt and .jpg files.
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} is not a directory")

    # Get a list of all items in the directory
    items = os.listdir(root_dir)
    items.sort()

    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)

        # Ignore .txt and .jpg files
        if item.endswith('.txt') or item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.cache'):
            continue

        is_last = index == len(items) - 1
        
        if is_last:
            connector = "└──"
            new_prefix = prefix + "    "
        else:
            connector = "├──"
            new_prefix = prefix + "│   "

        print(f"{prefix}{connector} {item}")
        
        if os.path.isdir(path):
            print_directory_structure(path, new_prefix)

# Example usage:
root_directory = 'dataset'
print_directory_structure(root_directory)
