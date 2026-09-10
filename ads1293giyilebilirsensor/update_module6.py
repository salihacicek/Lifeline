import re

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/ads1293_pipeline/module6_training.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Replace num_classes=5 with num_classes=7
code = code.replace("num_classes=5", "num_classes=7")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("module6 updated successfully.")
