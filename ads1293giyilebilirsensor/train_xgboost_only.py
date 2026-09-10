import sys
import pickle
import numpy as np
import scipy.signal
from ads1293_pipeline.module5_deep_learning import XGBoostClassifier
from train_ultimate_cardiologist import fetch_from_physionet, get_ptbxl_patients, fetch_from_ptbxl_local

def main():
    print("XGBoost Veri Toplama Başlıyor...", flush=True)
    all_cnn_x = []
    all_labels = []
    
    physionet_records = [
        ("mitdb", "100", "NORM"), ("mitdb", "101", "NORM"), ("mitdb", "103", "NORM"),
        ("mitdb", "109", "L"), ("mitdb", "111", "L"), ("mitdb", "207", "L"),
        ("mitdb", "118", "R"), ("mitdb", "124", "R"), ("mitdb", "212", "R"),
        ("mitdb", "106", "V"), ("mitdb", "119", "V"), ("mitdb", "200", "V"),
        ("mitdb", "209", "A"), ("mitdb", "220", "A"), ("mitdb", "223", "A")
    ]
    for db, rec, cls in physionet_records:
        x, y = fetch_from_physionet(db, rec, cls, duration=60)
        for segment in x:
            if len(segment) != 250: segment = scipy.signal.resample(segment, 250)
            all_cnn_x.append(segment)
            all_labels.append(y[0]) # y is a list of same class
            
    print("MIT-BIH bitti.", flush=True)
    
    ptbxl_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    records_dict = get_ptbxl_patients(ptbxl_path, num_patients_per_class=50)
    for superclass, patients in records_dict.items():
        for rec in patients:
            x, y = fetch_from_ptbxl_local(ptbxl_path, rec, superclass)
            for segment in x:
                if len(segment) != 250: segment = scipy.signal.resample(segment, 250)
                all_cnn_x.append(segment)
                all_labels.append(y[0])
                
    print(f"Toplam Vuruş: {len(all_cnn_x)}", flush=True)
    
    # Extract XGB features directly without keeping all_cnn_x in memory if we wanted, but it's fine.
    print("XGBoost özellikleri çıkarılıyor...", flush=True)
    xgb_train_x = np.array([[np.mean(beat), np.std(beat), np.max(beat), np.min(beat)] for beat in all_cnn_x])
    y_train = np.array(all_labels)
    
    # Free memory
    del all_cnn_x
    
    print("XGBoost eğitiliyor...", flush=True)
    xgb_model = XGBoostClassifier(num_classes=9, n_estimators=100)
    xgb_model.train(xgb_train_x, y_train)
    
    with open("xgboost_weights_9class.pkl", "wb") as f:
        pickle.dump(xgb_model.model, f)
        
    print("XGBoost başarıyla kaydedildi!", flush=True)

if __name__ == "__main__":
    main()
