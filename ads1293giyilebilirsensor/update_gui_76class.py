import os
import pandas as pd

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Add pandas import if not exists
if "import pandas as pd" not in code:
    code = code.replace("import numpy as np", "import numpy as np\nimport pandas as pd")

# Add the dynamic clinical logic

old_logic = """        classes = {
            0: ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676"),
            1: ("LBBB (SOL DAL BLOĞU - Aritmi)", "#ff9800"), 
            2: ("RBBB (SAĞ DAL BLOĞU - Aritmi)", "#ff9800"),
            3: ("PVC (ERKEN KARINCIK VURUSU)", "#f44336"), 
            4: ("APC (ERKEN KULAKÇIK VURUSU)", "#f44336"),
            5: ("MI (MİYOKARD ENFARKTÜSÜ - KRİZ)", "#9c27b0"),
            6: ("STTC (İSKEMİ / ST-T DEĞİŞİMİ)", "#e91e63"),
            7: ("CD (İLETİM BOZUKLUĞU / BLOK)", "#ff5722"),
            8: ("HYP (HİPERTROFİ / KALP BÜYÜMESİ)", "#673ab7")
        }
        
        label_text, color = classes.get(pred_class, ("BİLİNMİYOR", "#ffffff"))
        
        self.lbl_ai.setText(f"{label_text}\\n(Hibrit Yapay Zeka Güveni: %{confidence:.1f})")
        self.lbl_ai.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")"""

new_logic = """        # 76 Sınıf Sözlüğü Otomatik Oluşturuluyor
        scp_df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv", index_col=0)
        PTBXL_CLASSES = list(scp_df.index)
        MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
        ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES
        
        pred_class_name = ALL_CLASSES[pred_class]
        
        clinical_decision = ""
        color = "#ffffff"
        
        if pred_class_name in MIT_CLASSES:
            mit_dict = {
                "NORM_MIT": ("NORMAL SİNÜS RİTMİ (Sağlıklı)", "#00e676", "Gözlemlenecek bulgu yok. Sağlıklı beslenme ve düzenli spora devam ediniz."),
                "L_MIT": ("LBBB (Sol Dal Bloğu)", "#ff9800", "Ritim takibi yapılmalı, EKO önerilir."),
                "R_MIT": ("RBBB (Sağ Dal Bloğu)", "#ff9800", "İzlenmeli, nadiren tedavi gerektirir."),
                "V_MIT": ("PVC (Erken Karıncık Vurusu)", "#f44336", "Kardiyoloji kontrolü önerilir, ritim holteri takılabilir."),
                "A_MIT": ("APC (Erken Kulakçık Vurusu)", "#f44336", "Çarpıntı varsa ilaç tedavisi başlanabilir.")
            }
            label_text, color, clinical_decision = mit_dict.get(pred_class_name)
        else:
            row = scp_df.loc[pred_class_name]
            desc = row['description'].upper() if pd.notna(row['description']) else pred_class_name
            cat = row['diagnostic_class'] if pd.notna(row['diagnostic_class']) else "OTHER"
            
            if cat == "MI":
                label_text = f"{pred_class_name} - {desc} (KALP KRİZİ!)"
                color = "#9c27b0"
                clinical_decision = "ACİL MÜDAHALE! Anjiyografi veya Bypass cerrahisi değerlendirilmelidir."
            elif cat == "NORM":
                label_text = f"{pred_class_name} - {desc} (Sağlıklı)"
                color = "#00e676"
                clinical_decision = "Gözlemlenecek bulgu yok. Sağlıklı yaşama devam."
            elif cat == "STTC":
                label_text = f"{pred_class_name} - {desc} (İskemi)"
                color = "#e91e63"
                clinical_decision = "Damar tıkanıklığı şüphesi! Efor testi ve ilaç tedavisi önerilir."
            elif cat == "CD":
                label_text = f"{pred_class_name} - {desc} (İletim Bloğu)"
                color = "#ff5722"
                clinical_decision = "Kardiyak pacemaker (pil) veya EPS (Elektrofizyolojik Çalışma) değerlendirilmeli."
            elif cat == "HYP":
                label_text = f"{pred_class_name} - {desc} (Kalp Büyümesi)"
                color = "#673ab7"
                clinical_decision = "Tansiyon kontrolü ve EKO ile kalp duvar kalınlığı takibi yapılmalı."
            else:
                label_text = f"{pred_class_name} - {desc}"
                color = "#03a9f4"
                clinical_decision = "Poliklinik kontrolü tavsiye edilir."
        
        self.lbl_ai.setText(f"{label_text}\\n(Güven: %{confidence:.1f})\\nKlinik Karar: {clinical_decision}")
        self.lbl_ai.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")"""

code = code.replace(old_logic, new_logic)


# Fix the class counting logic
old_count_logic = """        self.beat_counts[pred_class] += 1
        total_beats = sum(self.beat_counts.values())
        normal_count = self.beat_counts[0]
        abnormal_count = total_beats - normal_count"""

new_count_logic = """        self.beat_counts[pred_class] += 1
        total_beats = sum(self.beat_counts.values())
        # NORM_MIT ve NORM classlarını bul (0 ve 71 değil, dinamik)
        normal_idx = [i for i, c in enumerate(ALL_CLASSES) if "NORM" in c]
        normal_count = sum(self.beat_counts[i] for i in normal_idx)
        abnormal_count = total_beats - normal_count"""

code = code.replace(old_count_logic, new_count_logic)


old_summary_logic = """            # Hastalık isimlerini topla
            detected = []
            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")
            if self.beat_counts[5] > 0: detected.append("MI")
            if self.beat_counts[6] > 0: detected.append("İskemi(STTC)")
            if self.beat_counts[7] > 0: detected.append("İletim(CD)")
            if self.beat_counts[8] > 0: detected.append("Hipertrofi(HYP)")
                
            summary = f"Kayıt Süresi: {time_str}{warning}  |  İncelenen Atım: {total_beats}  |  Normal: {normal_count}  |  Anormal: {abnormal_count}\\nNihai Sonuç: {risk_text}{disease_str}"
            self.lbl_summary.setText(summary)
            self.lbl_summary.setStyleSheet(f"font-size: 18px; color: {risk_color}; font-weight: bold;")"""

new_summary_logic = """            # Hastalık isimlerini topla
            detected = []
            for idx, count in self.beat_counts.items():
                if count > 0 and idx not in normal_idx:
                    detected.append(ALL_CLASSES[idx])
            disease_str = f" -> Hastalıklar: {', '.join(detected)}" if detected else ""
                
            summary = f"Süre: {time_str}{warning}  |  Atım: {total_beats}  |  Normal: {normal_count}  |  Anormal: {abnormal_count}\\nNihai Sonuç: {risk_text}{disease_str}"
            self.lbl_summary.setText(summary)
            self.lbl_summary.setStyleSheet(f"font-size: 18px; color: {risk_color}; font-weight: bold;")"""

code = code.replace(old_summary_logic, new_summary_logic)


# Update model weights load block
old_weight_block = """        # Eğitilmiş gerçek model ağırlıklarını (varsa) yükle
        weights_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights_9class.pth')
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print("✅ DERİN ÖĞRENME MODELİ (CNN) YÜKLENDİ (9 Sınıf)!")
            except Exception as e:
                print("CNN Model yükleme hatası:", e)
                
        self.model.eval()

        # XGBoost Modelini yükle
        xgb_path = os.path.join(os.path.dirname(__file__), 'xgboost_weights_9class.pkl')
        self.xgb_model = None
        if os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print("✅ MAKİNE ÖĞRENMESİ MODELİ (XGBOOST) YÜKLENDİ! (Hibrit Sistem Devrede)")
            except Exception as e:
                print("XGBoost Model yükleme hatası:", e)"""

new_weight_block = """        # Eğitilmiş gerçek model ağırlıklarını (varsa) yükle
        self.model = ECGHybridModel(in_channels=1, num_classes=76).to(self.device)
        weights_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights_76class.pth')
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print("✅ DERİN ÖĞRENME MODELİ (CNN) YÜKLENDİ (76 Sınıf - Mega Model)!")
            except Exception as e:
                print("CNN Model yükleme hatası:", e)
                
        self.model.eval()

        # XGBoost Modelini yükle
        xgb_path = os.path.join(os.path.dirname(__file__), 'xgboost_weights_76class.pkl')
        self.xgb_model = None
        if os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print("✅ MAKİNE ÖĞRENMESİ MODELİ (XGBOOST) YÜKLENDİ! (76 Sınıf)")
            except Exception as e:
                print("XGBoost Model yükleme hatası:", e)"""

code = code.replace("self.model = ECGHybridModel(in_channels=1, num_classes=9).to(self.device)\n" + old_weight_block, new_weight_block)

# Add NORM to the dropdown
code = code.replace("self.record_selector.addItems([", "self.record_selector.addItems([\n            'PTB-XL: 00001_lr (Sağlıklı - NORM)',\n            'PTB-XL: 00002_lr (Sağlıklı - NORM)',\n")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
