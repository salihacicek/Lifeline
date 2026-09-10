import re

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/ads1293_pipeline/module5_deep_learning.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Replace num_classes=5 with num_classes=7
code = code.replace("num_classes: int = 5", "num_classes: int = 7")

# Replace CLASS_NAMES
old_classes = 'CLASS_NAMES: List[str] = ["Normal", "LBBB", "RBBB", "PVC", "APC"]'
new_classes = 'CLASS_NAMES: List[str] = ["Normal", "LBBB", "RBBB", "PVC", "APC", "AF", "MI"]'
code = code.replace(old_classes, new_classes)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("module5 updated successfully.")
