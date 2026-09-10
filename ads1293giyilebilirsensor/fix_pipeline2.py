import os
for fname in ["train_7class.py", "train_hybrid_sources.py"]:
    filepath = os.path.join("/Users/salihacicek/Desktop/ads1293giyilebilirsensor", fname)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        code = code.replace("cleaned_signal, _ = pipeline2.run(raw_signal)", "cleaned_signal = pipeline2.run(raw_signal)['ecg_cleaned']")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
