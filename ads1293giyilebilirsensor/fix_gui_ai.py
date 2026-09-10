import os
filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/gui_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

bad_block = """        # Eğitilmiş gerçek model ağırlıklarını (varsa) yükle
            try:
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print("✅ MAKİNE ÖĞRENMESİ MODELİ (XGBOOST) YÜKLENDİ! (Hibrit Sistem Devrede)")
            except Exception as e:
                print("XGBoost Model yükleme hatası:", e)"""

good_block = """        # Eğitilmiş gerçek model ağırlıklarını (varsa) yükle
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

code = code.replace(bad_block, good_block)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
