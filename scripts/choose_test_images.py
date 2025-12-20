import torch
from rfdetr import RFDETRMedium
import os
import torch.nn as nn
import supervision as sv
import cv2

# 1. Inicializace základního wrapperu
model_wrapper = RFDETRMedium()

# 2. RUČNÍ PŘENASTAVENÍ POČTU TŘÍD (Z 91 NA 3)
# Musíme změnit všechny vrstvy, které si stěžovaly na mismatch
def fix_channels(module):
    if isinstance(module, nn.Linear) and module.out_features == 91:
        new_layer = nn.Linear(module.in_features, 3)
        return new_layer
    return module

# Projdeme skutečný vnitřní PyTorch model
# Přidali jsme .model za model_wrapper.model
inner_model = model_wrapper.model.model 

for name, m in inner_model.named_modules():
    if 'class_embed' in name or 'enc_out_class_embed' in name:
        if isinstance(m, nn.Linear) and m.out_features == 91:
            # Získáme cestu k atributu a nahradíme ho
            parts = name.split('.')
            obj = inner_model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], nn.Linear(m.in_features, 3))

print("🔧 Model v paměti byl přenastaven na 3 třídy.")

# 3. Načtení tvých vah (změna cesty k load_state_dict)
checkpoint = torch.load('03_trenovany_model/checkpoint_best_total.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint

# Načítáme přímo do inner_model
inner_model.load_state_dict(state_dict)
inner_model.eval()
print("✅ Váhy byly úspěšně načteny!")

# 4. Generování screenshotů
test_path = '02_dataset_coco/test'
output_path = '04_zaverecna_zprava/predikce'
os.makedirs(output_path, exist_ok=True)

# Inicializace anotátorů ze sady Supervision
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

images = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png'))][:5]
print("🎨 Kreslím detekce do obrázků...")

for img_name in images:
    img_full_path = os.path.join(test_path, img_name)
    
    # 1. Načtení obrázku pomocí OpenCV
    image = cv2.imread(img_full_path)
    
    # 2. Získání detekcí
    res = model_wrapper.predict(img_full_path, conf_threshold=0.5)
    
    # 3. Vykreslení boxů a popisků do obrázku
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=res)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=res)
    
    # 4. Uložení výsledku
    output_file = os.path.join(output_path, f'predikce_{img_name}')
    cv2.imwrite(output_file, annotated_image)
    
    print(f"📸 Hotovo: {img_name}")

print(f"\n🚀 Hotovo! Výsledky najdeš v: {os.path.abspath(output_path)}")