import asyncio
import json
import numpy as np
import os
import time
import torch
import scipy.signal
import wfdb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

def sanitize(val, default):
    return float(val) if not math.isnan(float(val)) and not math.isinf(float(val)) else default

from ads1293_pipeline.module1_data_acquisition import RawDataPipeline, ADCScaler
from ads1293_pipeline.module2_signal_processing import SignalCleaningPipeline
from ads1293_pipeline.module3_feature_extraction import PanTompkinsDetector, FeatureExtractionPipeline, PQRSTExtractor, HRVAnalyzer
from ads1293_pipeline.module3_feature_extraction import PQRSTExtractor
from ads1293_pipeline.module5_deep_learning import ECGHybridModel
from ads1293_pipeline.module8_xai import GradCAMExplainer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ECGHybridModel(in_channels=1, num_classes=15).to(device)
try:
    model.load_state_dict(torch.load("ecg_model_weights_15class.pth", map_location=device, weights_only=True))
    model.eval()
    print("15-Class ECG Model loaded.")
    explainer = GradCAMExplainer(model, model.cnn.branch_sharp[0])
    print("Grad-CAM Explainer initialized.")
except Exception as e:
    print("Could not load model:", e)

@app.get("/api/files")
async def list_files():
    # Return available simulation files like in the GUI dropdown
    full_list = [
        'MIT-BIH (Gerçek Hasta): 100', 'MIT-BIH (Gerçek Hasta): 101', 'MIT-BIH (Gerçek Hasta): 102', 'MIT-BIH (Gerçek Hasta): 103', 'MIT-BIH (Gerçek Hasta): 104', 'MIT-BIH (Gerçek Hasta): 105', 'MIT-BIH (Gerçek Hasta): 106', 'MIT-BIH (Gerçek Hasta): 107', 'MIT-BIH (Gerçek Hasta): 108', 'MIT-BIH (Gerçek Hasta): 109', 
        'MIT-BIH (Gerçek Hasta): 111', 'MIT-BIH (Gerçek Hasta): 112', 'MIT-BIH (Gerçek Hasta): 113', 'MIT-BIH (Gerçek Hasta): 114', 'MIT-BIH (Gerçek Hasta): 115', 'MIT-BIH (Gerçek Hasta): 116', 'MIT-BIH (Gerçek Hasta): 117', 'MIT-BIH (Gerçek Hasta): 118', 'MIT-BIH (Gerçek Hasta): 119', 'MIT-BIH (Gerçek Hasta): 121', 
        'MIT-BIH (Gerçek Hasta): 122', 'MIT-BIH (Gerçek Hasta): 123', 'MIT-BIH (Gerçek Hasta): 124', 'MIT-BIH (Gerçek Hasta): 200', 'MIT-BIH (Gerçek Hasta): 201', 'MIT-BIH (Gerçek Hasta): 202', 'MIT-BIH (Gerçek Hasta): 203', 'MIT-BIH (Gerçek Hasta): 205', 'MIT-BIH (Gerçek Hasta): 207', 'MIT-BIH (Gerçek Hasta): 208', 
        'MIT-BIH (Gerçek Hasta): 209', 'MIT-BIH (Gerçek Hasta): 210', 'MIT-BIH (Gerçek Hasta): 212', 'MIT-BIH (Gerçek Hasta): 213', 'MIT-BIH (Gerçek Hasta): 214', 'MIT-BIH (Gerçek Hasta): 215', 'MIT-BIH (Gerçek Hasta): 217', 'MIT-BIH (Gerçek Hasta): 219', 'MIT-BIH (Gerçek Hasta): 220', 'MIT-BIH (Gerçek Hasta): 221',
        'PTB-XL (Gerçek Hasta): records100/00000/00001_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00002_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00003_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00004_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00005_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00006_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00007_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00008_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00009_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00010_lr', 
        'PTB-XL (Gerçek Hasta): records100/00000/00011_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00012_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00013_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00014_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00015_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00016_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00017_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00018_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00019_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00020_lr', 
        'PTB-XL (Gerçek Hasta): records100/00000/00021_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00022_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00023_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00024_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00025_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00026_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00027_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00028_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00029_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00030_lr', 
        'PTB-XL (Gerçek Hasta): records100/00000/00031_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00032_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00033_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00034_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00035_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00036_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00037_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00038_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00039_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00040_lr'
    ]
    return {"status": "success", "files": full_list}

class ReportRequest(BaseModel):
    bpm: float
    qrs: float
    pr: float
    qt: float
    diagnosis: str
    confidence: float
    total_beats: int
    abnormal_count: int

@app.post("/api/generate_report")
async def generate_report(req: ReportRequest):
    import random
    
    # Simulate LLM thinking delay
    await asyncio.sleep(random.uniform(1.0, 2.0))
    
    # Generate dynamic report
    intro_opts = [
        "Hastanın sürekli EKG izleminden elde edilen son verilere göre,",
        "Klinik karar destek sisteminin anlık analiz sonuçları incelendiğinde,",
        "Giyilebilir sensör aracılığıyla kaydedilen ritim holter verilerinde,",
        "Yapay zeka tabanlı sinyal işleme algoritmalarımız sonucunda,"
    ]
    
    bpm_text = f"ortalama kalp hızı {req.bpm:.0f} BPM (Atım/Dakika) olarak ölçülmüştür."
    if req.bpm > 100:
        bpm_text = f"kalp hızında belirgin artış (Taşikardi: {req.bpm:.0f} BPM) saptanmıştır."
    elif req.bpm < 60:
        bpm_text = f"kalp hızında yavaşlama (Bradikardi: {req.bpm:.0f} BPM) izlenmiştir."
        
    morph_opts = [
        f"Morfolojik analizde QRS genişliği {req.qrs:.1f} ms, PR aralığı {req.pr:.1f} ms ve QT aralığı {req.qt:.1f} ms olarak hesaplanmıştır.",
        f"Elektrokardiyografik zaman serisi analizinde; QRS süresi ({req.qrs:.1f} ms), PR ({req.pr:.1f} ms) ve QT ({req.qt:.1f} ms) değerleri tespit edilmiştir."
    ]
    
    abn_text = f"Toplam incelenen {req.total_beats} atımın {req.abnormal_count} tanesinde anormallik gözlemlenmiştir."
    if req.abnormal_count == 0:
        abn_text = f"İncelenen {req.total_beats} atımın tamamı normal sinüs ritmi karakteristiğindedir."
    
    diag_opts = [
        f"Derin öğrenme modeli (CNN-BiLSTM) tarafından **% {req.confidence:.1f}** güven skoru ile **{req.diagnosis}** tanısı konulmuştur.",
        f"Yapay zeka analiz motoru, mevcut EKG dalga formunu **{req.diagnosis}** (Güven: %{req.confidence:.1f}) olarak sınıflandırmıştır."
    ]
    
    conc_opts = [
        "Bu bulgular ışığında, hastanın uzman bir kardiyolog tarafından değerlendirilmesi önerilir.",
        "Rapor edilen tanı ve metriklerin klinik korelasyonunun yapılması tavsiye edilir.",
        "Herhangi bir müdahale öncesinde sistem bulgularının hekim tarafından doğrulanması esastır.",
        "Hastanın vital bulgularının takibine devam edilmelidir."
    ]
    if "NORMAL" in req.diagnosis.upper():
        conc_opts = [
            "Mevcut EKG bulguları fizyolojik sınırlar içerisindedir, akut kardiyak patoloji saptanmamıştır.",
            "Hastanın ritim verilerinde anormallik bulunmamaktadır, rutin kontrollere devam edilebilir."
        ]
        
    report = f"{random.choice(intro_opts)} {bpm_text} {random.choice(morph_opts)} {abn_text} {random.choice(diag_opts)} {random.choice(conc_opts)}"
    
    return {"status": "success", "report": report}

@app.websocket("/ws/ecg/{filename:path}")
async def websocket_ecg_endpoint(websocket: WebSocket, filename: str):
    await websocket.accept()
    
    fs = 360 # Standard GUI rate
    
    # 1. Fetch data logic (similar to fetch_and_start in gui_app.py)
    try:
        if filename.startswith("PTB-XL"):
            record_filename = filename.split(": ")[1]
            if " (" in record_filename:
                record_filename = record_filename.split(" ")[0]
            ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
            record_path = os.path.join(ptbxl_base_path, record_filename)
            
            record = wfdb.rdrecord(record_path, channels=[0])
            ecg_mv = record.p_signal[:, 0]
            orig_fs = record.fs
            if orig_fs != fs:
                num_samples = int(len(ecg_mv) * fs / orig_fs)
                ecg_mv = scipy.signal.resample(ecg_mv, num_samples)
            sim_data = ecg_mv * 1000.0
        elif filename.startswith("MIT-BIH"):
            rec_id = filename.split(": ")[1]
            pipeline = RawDataPipeline(record_id=rec_id)
            data = pipeline.fetch_data(duration_seconds=300)
            sim_data = data["ecg_uv"]
        else:
            sim_data = np.zeros(fs * 10)
    except Exception as e:
        print(f"Error fetching data: {e}")
        await websocket.close(code=1011)
        return

    idx = 0
    start_time = time.time()
    
    clean_pipeline = SignalCleaningPipeline(fs=fs)
    pt_detector = PanTompkinsDetector(fs=fs)
    hrv_analyzer = HRVAnalyzer()
    pqrst_extractor = PQRSTExtractor(fs=fs)
    
    window_seconds = 5
    buffer_size = window_seconds * fs
    raw_data_buffer = [0.0] * buffer_size
    
    beat_counts = {i: 0 for i in range(15)}
    
    try:
        while True:
            # Emulate streaming
            if idx >= len(sim_data):
                idx = 0 # Loop the signal
                
            val = sim_data[idx]
            raw_data_buffer.pop(0)
            raw_data_buffer.append(val)
            
            # Every 1 second (360 samples), run AI logic
            ai_payload = None
            if idx > 0 and idx % fs == 0:
                raw_segment = np.array(raw_data_buffer)
                if not np.all(raw_segment == 0):
                    try:
                        clean_res = clean_pipeline.run(raw_segment, normalize=True)
                        clean_data_buffer = clean_res["ecg_normalized"]
                        r_peaks = pt_detector.detect(clean_data_buffer)
                        
                        bpm, qrs, pr, qt = 0, np.nan, np.nan, np.nan
                        if len(r_peaks) >= 2:
                            rr_intervals = pt_detector.get_rr_intervals(r_peaks)
                            bpm = hrv_analyzer.compute_time_domain(rr_intervals).get("mean_hr_bpm", 0)
                            
                            pqrst_list = []
                            for peak in r_peaks:
                                waves = pqrst_extractor.extract_waves(clean_data_buffer, peak)
                                intervals = pqrst_extractor.compute_intervals(waves)
                                pqrst_list.append(intervals)
                                
                            if pqrst_list:
                                qrs = np.nanmean([d["QRS_Width_ms"] for d in pqrst_list])
                                pr = np.nanmean([d["PR_Interval_ms"] for d in pqrst_list])
                                qt = np.nanmean([d["QT_Interval_ms"] for d in pqrst_list])
                                
                            rp = r_peaks[len(r_peaks)//2]
                            start_idx_beat = rp - int(fs * 0.3)
                            end_idx_beat = rp + int(fs * 0.5)
                            
                            if start_idx_beat >= 0 and end_idx_beat <= len(clean_data_buffer):
                                beat = clean_data_buffer[start_idx_beat:end_idx_beat]
                                if len(beat) != 250:
                                    beat = scipy.signal.resample(beat, 250)
                                input_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                                
                                torch.set_grad_enabled(True)
                                
                                logits = model(input_tensor)
                                probs = torch.nn.functional.softmax(logits, dim=1)[0].detach().cpu().numpy()
                                pred_class = np.argmax(probs)
                                confidence = probs[pred_class] * 100
                                
                                # Heatmap hesaplama
                                heatmap = explainer.explain(input_tensor, int(pred_class))
                                heatmap_list = heatmap.tolist()
                                beat_list = beat.tolist()
                                
                                torch.set_grad_enabled(False)
                                
                                beat_counts[pred_class] += 1
                                
                                pred_class_name = ALL_CLASSES[pred_class]
                                label_text, color, clinical_decision = dict_15.get(pred_class_name, ("BİLİNMİYOR", "#ffffff", "Doktora danışın."))
                                
                                total_beats = sum(beat_counts.values())
                                normal_idx = [i for i, c in enumerate(ALL_CLASSES) if "NORM" in c]
                                normal_count = sum(beat_counts[i] for i in normal_idx)
                                abnormal_count = total_beats - normal_count
                                
                                ai_payload = {
                                    "bpm": bpm,
                                    "qrs": qrs,
                                    "pr": pr,
                                    "qt": qt,
                                    "diagnosis": label_text,
                                    "confidence": confidence,
                                    "color": color,
                                    "clinical_decision": clinical_decision,
                                    "total_beats": total_beats,
                                    "normal_count": normal_count,
                                    "abnormal_count": abnormal_count,
                                    "beat_counts": beat_counts,
                                    "heatmap": heatmap_list,
                                    "heatmap_signal": beat_list
                                }
                    except Exception as e:
                        print("AI Error:", e)

            # Send data frame
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)

            payload = {
                "idx": idx,
                "time": float(idx / fs),
                "raw_point": sanitize(val, 0.0),
                "record_time": f"{mins:02}:{secs:02}"
            }
            if ai_payload:
                payload["ai"] = {
                    "bpm": sanitize(ai_payload["bpm"], 0),
                    "qrs": sanitize(ai_payload["qrs"], 0),
                    "pr": sanitize(ai_payload["pr"], 0),
                    "qt": sanitize(ai_payload["qt"], 0),
                    "diagnosis": ai_payload["diagnosis"],
                    "confidence": sanitize(ai_payload["confidence"], 0),
                    "color": ai_payload["color"],
                    "clinical_decision": ai_payload["clinical_decision"],
                    "total_beats": ai_payload["total_beats"],
                    "normal_count": ai_payload["normal_count"],
                    "abnormal_count": ai_payload["abnormal_count"],
                    "beat_counts": ai_payload["beat_counts"],
                    "heatmap": ai_payload.get("heatmap", []),
                    "heatmap_signal": ai_payload.get("heatmap_signal", [])
                }
                
            await websocket.send_json(payload)
            idx += 1
            
            # 360 Hz -> 2.77ms sleep
            await asyncio.sleep(1/fs)
            
    except (WebSocketDisconnect, RuntimeError):
        print(f"Client disconnected")
    except Exception as e:
        print(f"Unexpected websocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
