import json
import os
import random
import shutil

def split_coco(json_path, image_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    random.shuffle(images)

    # Calculate split points
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    # Process each split
    for name, split_images in splits.items():
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        
        img_ids = {img['id'] for img in split_images}
        split_anns = [ann for ann in data['annotations'] if ann['image_id'] in img_ids]
        
        # Build new JSON
        split_coco = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": data.get("categories", []),
            "images": split_images,
            "annotations": split_anns
        }
        
        # Save JSON
        with open(os.path.join(output_dir, f"{name}_annotations.json"), 'w') as f:
            json.dump(split_coco, f)
            
        # Move Images
        print(f"Moving {len(split_images)} images to {name}...")
        for img in split_images:
            shutil.copy(os.path.join(image_dir, img['file_name']), 
                        os.path.join(split_dir, img['file_name']))

    print("✨ Dataset split 80/10/10 successfully!")

# Usage
split_coco(
    json_path='instances_default.json', # Your exported CVAT filename
    image_dir='images/',                # Where CVAT exported the jpgs
    output_dir='final_dataset/'
)