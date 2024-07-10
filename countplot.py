import os
import matplotlib.pyplot as plt

# Dictionary mapping class IDs to class names
class_names = {
    0: 'car', 1: 'bike', 2: 'auto', 3: 'rickshaw', 4: 'cycle',
    5: 'bus', 6: 'minitruck', 7: 'truck', 8: 'van', 9: 'taxi',
    10: 'motorvan', 11: 'toto', 12: 'train', 13: 'boat', 14: 'other'
}

def count_classes_in_folder(folder_path):
    class_counts = {class_id: 0 for class_id in class_names.keys()}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                annotations = f.readlines()
                for line in annotations:
                    try:
                        class_id = int(line.split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                    except ValueError:
                        continue  # Skip lines that cannot be converted to int
    
    return class_counts

def plot_class_counts(class_counts, folder_name, save_path=None):
    classes = [class_names[class_id] for class_id in sorted(class_counts.keys())]
    counts = [class_counts[class_id] for class_id in sorted(class_counts.keys())]
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title(f'Class Counts in {folder_name} folder')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, f'class_counts_{folder_name.lower()}.png')
        plt.savefig(save_file)
        print(f"Countplot saved as {save_file}")
    else:
        plt.show()

def main(dataset_root_dir):
    label_train_dir = os.path.join(dataset_root_dir, 'labels', 'train')
    label_val_dir = os.path.join(dataset_root_dir, 'labels', 'val')
    
    train_class_counts = count_classes_in_folder(label_train_dir)
    val_class_counts = count_classes_in_folder(label_val_dir)
    
    plot_class_counts(train_class_counts, 'Train', save_path='graphs/')
    plot_class_counts(val_class_counts, 'Validation', save_path='graphs/')

if __name__ == "__main__":
    dataset_root_dir = 'dataset'  # Adjust this path according to your dataset structure
    main(dataset_root_dir)
