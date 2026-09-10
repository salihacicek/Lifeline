import os

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update the ALL_CLASSES
old_classes = """        scp_df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv", index_col=0)
        PTBXL_CLASSES = list(scp_df.index)
        MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
        ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES"""

new_classes = """        PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
        MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
        ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES"""

code = code.replace(old_classes, new_classes)

# 2. Update the logic for fetching clinical decision
old_clinical = """        if pred_class_name in MIT_CLASSES:
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
                clinical_decision = "Poliklinik kontrolü tavsiye edilir.""""

new_clinical = """        # Gerçek 15 Sınıf Klinik Karar Mekanizması
        dict_15 = {
            "NORM_MIT": ("NORMAL SİNÜS RİTMİ", "#00e676", "Sağlıklı Ritim. Spora ve düzenli beslenmeye devam edin."),
            "L_MIT": ("LBBB (Sol Dal Bloğu)", "#ff9800", "Sol dal bloğu şüphesi. Kalp yetmezliği açısından EKO istenmeli."),
            "R_MIT": ("RBBB (Sağ Dal Bloğu)", "#ff9800", "Sağ dal bloğu. Çoğunlukla zararsızdır ancak düzenli izlenmelidir."),
            "V_MIT": ("PVC (Erken Karıncık Vurusu)", "#f44336", "Ektopik atım tespit edildi. Çarpıntı devam ederse Holter takılmalı."),
            "A_MIT": ("APC (Erken Kulakçık Vurusu)", "#f44336", "Erken atım saptandı. Aşırı çay/kahve tüketimi veya strese bağlı olabilir."),
            "NORM": ("NORMAL EKG", "#00e676", "Mükemmel Sağlıklı. EKG'de hiçbir problem tespit edilmedi."),
            "IMI": ("İnferiyor MİYOKARD ENFARKTÜSÜ", "#9c27b0", "ACİL! Alt duvar kalp krizi bulgusu! Hemen acile başvurunuz."),
            "ASMI": ("Anteroseptal MİYOKARD ENFARKTÜSÜ", "#9c27b0", "ACİL! Ön duvar kalp krizi bulgusu! Anjiyografi değerlendirilmelidir."),
            "LVH": ("SOL KARINCIK HİPERTROFİSİ", "#673ab7", "Kalp duvarında kalınlaşma (Kalp Büyümesi). Tansiyon takibi şarttır."),
            "LAFB": ("Sol Ön Dal Bloğu", "#ff5722", "İletim gecikmesi. Asemptomatik ise müdahale gerektirmeyebilir."),
            "1AVB": ("Birinci Derece AV Blok", "#ff5722", "Kulakçık ile karıncık arası iletim gecikmiş. İlaç veya pacemaker kontrolü."),
            "CRBBB": ("Tam Sağ Dal Bloğu", "#ff9800", "Sağ karıncıkta iletim blokajı. Yapısal kalp hastalığı araştırılmalı."),
            "CLBBB": ("Tam Sol Dal Bloğu", "#ff9800", "Sol karıncıkta iletim blokajı. Gizli iskemik kalp hastalığı riski."),
            "AFIB": ("ATRİYAL FİBRİLASYON", "#e91e63", "Düzensiz ve tehlikeli ritim! Pıhtı atma riski yüksek, kan sulandırıcı başlanmalı."),
            "STACH": ("SİNÜS TAŞİKARDİSİ", "#f44336", "Aşırı hızlı kalp atımı (BPM > 100). Efor, ateş, hipertiroid veya stres kaynaklı olabilir.")
        }
        label_text, color, clinical_decision = dict_15.get(pred_class_name, ("BİLİNMİYOR", "#ffffff", "Doktora danışın."))"""

code = code.replace(old_clinical, new_clinical)

# 3. Clean fetch_and_start from ground truth extraction
old_fetch_start = """        def fetch_and_start():
            try:
                self.ground_truth_classes = []
                if rec_id.startswith("PTB-XL"):
                    # ptbxl string looks like: PTB-XL (AF): records100/00000/00017_lr
                    record_filename = rec_id.split(": ")[1]
                    # Fix for healthy patients that have extra text in the filename string
                    if " (" in record_filename:
                        record_filename = record_filename.split(" ")[0]
                    
                    ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
                    record_path = os.path.join(ptbxl_base_path, record_filename)
                    
                    # Extract ground truth from CSV to help the 30-epoch model
                    import ast
                    try:
                        df_db = pd.read_csv(os.path.join(ptbxl_base_path, "ptbxl_database.csv"))
                        # record_filename is records100/00000/00001_lr
                        row = df_db[df_db['filename_lr'] == record_filename]
                        if not row.empty:
                            codes = ast.literal_eval(row.iloc[0]['scp_codes'])
                            self.ground_truth_classes = list(codes.keys())
                    except:
                        pass

                    record = wfdb.rdrecord(record_path, channels=[0])
                    ecg_mv = record.p_signal[:, 0]
                    # Convert to microvolts as the GUI uses ecg_uv
                    self.sim_data = ecg_mv * 1000.0
                else:
                    # İnternet üzerinden PhysioNet veri çekimi (MIT-BIH)
                    pipeline = RawDataPipeline(record_id=rec_id)
                    data = pipeline.fetch_data(duration_seconds=300)
                    
                    # MIT ground truth heuristic
                    if rec_id in ["100", "101", "103"]: self.ground_truth_classes = ["NORM_MIT"]
                    elif rec_id in ["109", "111", "207"]: self.ground_truth_classes = ["L_MIT"]
                    elif rec_id in ["118", "124", "212"]: self.ground_truth_classes = ["R_MIT"]
                    elif rec_id in ["106", "119", "200"]: self.ground_truth_classes = ["V_MIT"]
                    elif rec_id in ["209", "220", "223"]: self.ground_truth_classes = ["A_MIT"]

                    self.sim_data = data["ecg_uv"]"""

new_fetch_start = """        def fetch_and_start():
            try:
                if rec_id.startswith("PTB-XL"):
                    record_filename = rec_id.split(": ")[1]
                    if " (" in record_filename:
                        record_filename = record_filename.split(" ")[0]
                    
                    ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
                    record_path = os.path.join(ptbxl_base_path, record_filename)
                    
                    record = wfdb.rdrecord(record_path, channels=[0])
                    ecg_mv = record.p_signal[:, 0]
                    self.sim_data = ecg_mv * 1000.0
                else:
                    pipeline = RawDataPipeline(record_id=rec_id)
                    data = pipeline.fetch_data(duration_seconds=300)
                    self.sim_data = data["ecg_uv"]"""

code = code.replace(old_fetch_start, new_fetch_start)

# 4. Remove prediction boost
old_boost = """        # DEMO MODE: Yapay zekanın 76 sınıf gibi devasa bir uzayda 30 epochluk eğitimle 
        # karışmasını önlemek için simülasyondaki gerçek hasta verisiyle (Ground Truth) modeli destekle
        if hasattr(self, 'ground_truth_classes') and self.ground_truth_classes:
            for gt_class in self.ground_truth_classes:
                if gt_class in ALL_CLASSES:
                    idx = ALL_CLASSES.index(gt_class)
                    final_probs[idx] += 2.0  # Olasılığı yapay olarak artır
                    
        pred_class = np.argmax(final_probs)
        # Güven oranını mantıklı bir seviyeye çek
        confidence = min(99.9, (final_probs[pred_class] / np.sum(final_probs)) * 100 * 2)
        if confidence < 50: confidence += 40"""

new_boost = """        pred_class = np.argmax(final_probs)
        confidence = final_probs[pred_class] * 100"""

code = code.replace(old_boost, new_boost)

# 5. Fix weights and model num_classes
code = code.replace("self.model = ECGHybridModel(in_channels=1, num_classes=76).to(self.device)", "self.model = ECGHybridModel(in_channels=1, num_classes=15).to(self.device)")
code = code.replace("ecg_model_weights_76class.pth", "ecg_model_weights_15class.pth")
code = code.replace("xgboost_weights_76class.pkl", "xgboost_weights_15class.pkl")
code = code.replace("76 Sınıf - Mega Model", "15 Sınıf - Gerçek AI")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

