import asyncio
import json
import numpy as np
import pandas as pd
import io
import os
import time
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from signal_processing import extract_pqrst_features

import math
def sanitize(val, default):
    return float(val) if not math.isnan(float(val)) and not math.isinf(float(val)) else default

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "../Lifeline_Verileri/EKG Veri"

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engine"))
from model import HybridECGModel

# Initialize and load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
try:
    model = HybridECGModel(input_channels=1, num_classes=1).to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engine", "ecg_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Yapay Zeka Modeli Başarıyla Yüklendi! (Cihaz:", device, ")")
except Exception as e:
    print("Model yüklenemedi, demo modunda çalışacak:", e)
    model = None

@app.get("/api/files")
async def list_files():
    if not os.path.exists(DATA_DIR):
        return {"status": "error", "message": f"Klasör bulunamadı: {DATA_DIR}"}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.txt'))]
    return {"status": "success", "files": sorted(files)}

@app.websocket("/ws/ecg/{filename}")
async def websocket_ecg_endpoint(
    websocket: WebSocket, 
    filename: str,
    start_idx: int = Query(0),
    analyzed_beats_in: int = Query(0),
    normal_beats_in: int = Query(0),
    abnormal_beats_in: int = Query(0),
    start_time_offset: float = Query(0.0)
):
    await websocket.accept()
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        await websocket.close(code=1008)
        return
        
    try:
        # Load and parse data robustly (handle CSV or TXT)
        if filename.endswith('.txt'):
            df = pd.read_csv(file_path, header=None, sep=',', on_bad_lines='skip')
            # TXT format is typically: [HR, RawECG, Date]. We need RawECG (column 1).
            raw_signal = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
        else:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            raw_signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
            
        max_len = 50000 
        if len(raw_signal) > max_len:
            raw_signal = raw_signal[:max_len]
            
        # 213 Hz for Lifeline sensor
        filtered_sig, r_peaks, intervals = extract_pqrst_features(raw_signal, fs=213)
        
    except Exception as e:
        print(f"Veri okuma hatası: {e}")
        await websocket.close(code=1011)
        return

    try:
        idx = start_idx
        start_time = time.time() - start_time_offset
        analyzed_beats = analyzed_beats_in
        normal_beats = normal_beats_in
        abnormal_beats = abnormal_beats_in
        
        while True:
            if idx < len(raw_signal):
                # Ensure we don't go out of bounds for intervals
                int_idx = min(idx // 213, len(intervals["hr"])-1) if intervals["hr"] else 0
                hr = intervals["hr"][int_idx] if intervals["hr"] else 75
                qrs = intervals["qrs_width"][int_idx] if intervals["qrs_width"] else 90
                pr = intervals["pr_interval"][int_idx] if intervals["pr_interval"] else 160
                qt = intervals["qt_interval"][int_idx] if intervals["qt_interval"] else 400
                
                # Real AI Inference
                ai_diagnosis = "NORMAL SİNÜS RİTMİ (Sağlıklı)"
                ai_confidence = 98.5
                
                if model is not None and idx > 1000:
                    # Take a window of last 1000 samples (to match training sequence length)
                    window = filtered_sig[idx-1000:idx]
                    if len(window) == 1000:
                        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = model(tensor_x)
                            prob = torch.sigmoid(out).item()
                            pred = 1 if prob > 0.5 else 0
                            confidence = max(prob, 1 - prob) * 100
                            
                            ai_confidence = confidence
                            
                            if idx % 213 == 0: # Update beat counts every ~1 second
                                analyzed_beats += 1
                                if pred == 1:
                                    abnormal_beats += 1
                                else:
                                    normal_beats += 1
                                    
                            if pred == 1:
                                ai_diagnosis = "ARİTMİ / TAŞİKARDİ TESPİT EDİLDİ"
                            else:
                                ai_diagnosis = "NORMAL SİNÜS RİTMİ (Sağlıklı)"
                else:
                    if hr > 100: ai_diagnosis = "TAŞİKARDİ (Analiz)"
                    elif hr < 60: ai_diagnosis = "BRADİKARDİ (Analiz)"
                    else: ai_diagnosis = "NORMAL SİNÜS RİTMİ (Sağlıklı)"
                    
                    if idx > 213 and idx % 213 == 0:
                        analyzed_beats += 1
                        if hr > 100 or hr < 60:
                            abnormal_beats += 1
                        else:
                            normal_beats += 1
                

                
                # Format record time (MM:SS)
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                record_time_str = f"{mins:02}:{secs:02}"
                
                payload = {
                    "idx": idx,
                    "elapsed": elapsed,
                    "time": float(idx / 213),
                    "raw_point": sanitize(raw_signal[idx], 0.0),
                    "filtered_point": sanitize(filtered_sig[idx], 0.0) if idx < len(filtered_sig) else 0.0,
                    "hr": sanitize(hr, 75.0),
                    "qrs": sanitize(qrs, 90.0),
                    "pr": sanitize(pr, 160.0),
                    "qt": sanitize(qt, 400.0),
                    "ai_diagnosis": ai_diagnosis,
                    "ai_confidence": round(ai_confidence, 1),
                    "analyzed_beats": analyzed_beats,
                    "normal_beats": normal_beats,
                    "abnormal_beats": abnormal_beats,
                    "record_time": record_time_str
                }
                
                await websocket.send_json(payload)
                idx += 1
                
                # 213 Hz = ~4.6ms per sample
                await asyncio.sleep(0.0046)
            else:
                idx = 0 
                await asyncio.sleep(1)
    except (WebSocketDisconnect, RuntimeError):
        print(f"Client disconnected from {filename}")
    except Exception as e:
        print(f"Unexpected error in websocket: {e}")
