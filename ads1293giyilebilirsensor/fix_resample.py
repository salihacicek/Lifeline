import os
filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/train_ultimate_cardiologist.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

import_scipy = "import scipy.signal\n"
if "scipy.signal" not in code:
    code = code.replace("import numpy as np", "import numpy as np\n" + import_scipy)

old_logic = """    cnn_X = []
    labels = []
    for segment in features.get("heartbeat_segments", []):
        cnn_X.append(segment)
        labels.append(target_class)
    return cnn_X, labels"""

new_logic = """    cnn_X = []
    labels = []
    for segment in features.get("heartbeat_segments", []):
        # Farklı FS (100 Hz vs 360 Hz) nedeniyle oluşan farklı uzunlukları sabitliyoruz
        if len(segment) != 250:
            segment = scipy.signal.resample(segment, 250)
        cnn_X.append(segment)
        labels.append(target_class)
    return cnn_X, labels"""

code = code.replace(old_logic, new_logic)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
