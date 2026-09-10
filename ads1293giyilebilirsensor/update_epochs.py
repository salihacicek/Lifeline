filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/train_hybrid_sources.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("epochs=3,", "epochs=30,")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
