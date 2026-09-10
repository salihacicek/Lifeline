import re

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Update model init
code = code.replace("num_classes=5", "num_classes=7")

# Update beat_counts initialization
code = code.replace("self.beat_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}", "self.beat_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}")

# Update classes dictionary
old_classes = """        classes = {
            0: ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676"),
            1: ("LBBB (SOL DAL BLOĞU - Aritmi)", "#ff9800"), 
            2: ("RBBB (SAĞ DAL BLOĞU - Aritmi)", "#ff9800"),
            3: ("PVC (ERKEN KARINCIK VURUSU - Aritmi)", "#f44336"), 
            4: ("APC (ERKEN KULAKÇIK VURUSU - Aritmi)", "#f44336")
        }"""
new_classes = """        classes = {
            0: ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676"),
            1: ("LBBB (SOL DAL BLOĞU - Aritmi)", "#ff9800"), 
            2: ("RBBB (SAĞ DAL BLOĞU - Aritmi)", "#ff9800"),
            3: ("PVC (ERKEN KARINCIK VURUSU - Aritmi)", "#f44336"), 
            4: ("APC (ERKEN KULAKÇIK VURUSU - Aritmi)", "#f44336"),
            5: ("AF (ATRİYAL FİBRİLASYON - Ritim)", "#e91e63"),
            6: ("ST+ (MİYOKARD ENFARKTÜSÜ - MI)", "#9c27b0")
        }"""
code = code.replace(old_classes, new_classes)

# Update detected appending logic
old_detected = """            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")"""
new_detected = """            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")
            if self.beat_counts[5] > 0: detected.append("AF")
            if self.beat_counts[6] > 0: detected.append("MI (ST+)")"""
code = code.replace(old_detected, new_detected)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("gui_app updated successfully.")
