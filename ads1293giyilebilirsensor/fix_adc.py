import os
filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/train_ultimate_cardiologist.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("scaler.ADC_MAX", "scaler.adc_max")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
