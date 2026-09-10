filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Replace classes dict
old_classes_dict = """        classes = {
            0: ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676"),
            1: ("LBBB (SOL DAL BLOĞU - Aritmi)", "#ff9800"), 
            2: ("RBBB (SAĞ DAL BLOĞU - Aritmi)", "#ff9800"),
            3: ("PVC (ERKEN KARINCIK VURUSU - Aritmi)", "#f44336"), 
            4: ("APC (ERKEN KULAKÇIK VURUSU - Aritmi)", "#f44336"),
            5: ("AF (ATRİYAL FİBRİLASYON - Ritim)", "#e91e63"),
            6: ("ST+ (MİYOKARD ENFARKTÜSÜ - MI)", "#9c27b0")
        }"""

new_classes_dict = """        classes = {
            0: ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676"),
            1: ("LBBB (SOL DAL BLOĞU - Aritmi)", "#ff9800"), 
            2: ("RBBB (SAĞ DAL BLOĞU - Aritmi)", "#ff9800"),
            3: ("PVC (ERKEN KARINCIK VURUSU)", "#f44336"), 
            4: ("APC (ERKEN KULAKÇIK VURUSU)", "#f44336"),
            5: ("MI (MİYOKARD ENFARKTÜSÜ - KRİZ)", "#9c27b0"),
            6: ("STTC (İSKEMİ / ST-T DEĞİŞİMİ)", "#e91e63"),
            7: ("CD (İLETİM BOZUKLUĞU / BLOK)", "#ff5722"),
            8: ("HYP (HİPERTROFİ / KALP BÜYÜMESİ)", "#673ab7")
        }"""

code = code.replace(old_classes_dict, new_classes_dict)

# Replace summary logic
old_summary_logic = """            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")
            if self.beat_counts[5] > 0: detected.append("AF")
            if self.beat_counts[6] > 0: detected.append("MI (ST+)")"""

new_summary_logic = """            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")
            if self.beat_counts[5] > 0: detected.append("MI")
            if self.beat_counts[6] > 0: detected.append("İskemi(STTC)")
            if self.beat_counts[7] > 0: detected.append("İletim(CD)")
            if self.beat_counts[8] > 0: detected.append("Hipertrofi(HYP)")"""

code = code.replace(old_summary_logic, new_summary_logic)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
