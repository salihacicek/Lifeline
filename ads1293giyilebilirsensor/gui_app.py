# -*- coding: utf-8 -*-
"""
=============================================================================
ADS1293 TELEMETRİ SİSTEMİ - CANLI İZLEME VE YAPAY ZEKA ARAYÜZÜ (GUI)
=============================================================================
Bu arayüz 3 farklı veri kaynağından (Simülasyon, Donanım, SD Kart) beslenebilir.
Ekranda görünen grafiklerin tamamı saf, gerçek insan kalbine ait (PhysioNet)
EKG verileridir.

Kullanım:
    python gui_app.py
=============================================================================
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from collections import deque
import threading
import time
import wfdb

# PyQT5 ve PyQtGraph
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QGroupBox, QRadioButton, QFileDialog,
                             QApplication, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

# Proje Dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ads1293_pipeline')))

from module1_data_acquisition import RawDataPipeline, OfflineDataPipeline
from module2_signal_processing import SignalCleaningPipeline
from module3_feature_extraction import PanTompkinsDetector, HRVAnalyzer, PQRSTExtractor
from module5_deep_learning import get_device, ECGHybridModel
from module9_hardware_serial import PortScanner, SerialDataReader
import torch
import pickle

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADS1293 EKG Telemetri - Hastane Monitörü")
        self.resize(1200, 850)
        
        # --- Veri Yapıları ---
        self.fs = 360 # Örnekleme frekansı (MIT-BIH Standardı)
        self.window_seconds = 5 # Ekranda gösterilecek saniye
        self.buffer_size = self.fs * self.window_seconds
        
        self.raw_data_buffer = deque([0.0]*self.buffer_size, maxlen=self.buffer_size)
        self.time_buffer = deque(np.linspace(-self.window_seconds, 0, self.buffer_size), maxlen=self.buffer_size)
        self.clean_data_buffer = np.zeros(self.buffer_size)
        
        self.is_running = False
        self.current_time = 0.0
        self.beat_counts = {i: 0 for i in range(15)} # Teşhis sayacı
        
        # --- Pipeline & AI Modelleri ---
        self.setup_ai_pipeline()
        
        # --- Modüller ---
        self.serial_reader = None
        self.simulator_thread = None
        self.sim_data = []
        self.sim_idx = 0
        self.sd_file_path = None
        
        # Standart MIT-BIH Hastaları
        self.patient_records = [
            # MIT-BIH (Aritmi) Kayıtları
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
            "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
            "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
            "222", "223", "228", "230", "231", "232", "233", "234",
            # PTB-XL (AF) Kayıtları
            'PTB-XL (AF): records100/00000/00017_lr', 'PTB-XL (AF): records100/00000/00152_lr', 
            'PTB-XL (AF): records100/00000/00282_lr', 'PTB-XL (AF): records100/00000/00307_lr', 
            'PTB-XL (AF): records100/00000/00318_lr', 'PTB-XL (AF): records100/00000/00321_lr', 
            'PTB-XL (AF): records100/00000/00330_lr', 'PTB-XL (AF): records100/00000/00337_lr', 
            'PTB-XL (AF): records100/00000/00351_lr', 'PTB-XL (AF): records100/00000/00428_lr', 
            'PTB-XL (AF): records100/00000/00452_lr', 'PTB-XL (AF): records100/00000/00482_lr', 
            'PTB-XL (AF): records100/00000/00500_lr', 'PTB-XL (AF): records100/00000/00547_lr', 
            'PTB-XL (AF): records100/00000/00557_lr', 'PTB-XL (AF): records100/00000/00567_lr', 
            'PTB-XL (AF): records100/00000/00569_lr', 'PTB-XL (AF): records100/00000/00581_lr', 
            'PTB-XL (AF): records100/00000/00594_lr', 'PTB-XL (AF): records100/00000/00598_lr',
            # PTB-XL (MI) Kayıtları
        ]

        self.sim_files = [
            'MIT-BIH (Gerçek Hasta): 100', 'MIT-BIH (Gerçek Hasta): 101', 'MIT-BIH (Gerçek Hasta): 102', 'MIT-BIH (Gerçek Hasta): 103', 'MIT-BIH (Gerçek Hasta): 104', 'MIT-BIH (Gerçek Hasta): 105', 'MIT-BIH (Gerçek Hasta): 106', 'MIT-BIH (Gerçek Hasta): 107', 'MIT-BIH (Gerçek Hasta): 108', 'MIT-BIH (Gerçek Hasta): 109', 
            'MIT-BIH (Gerçek Hasta): 111', 'MIT-BIH (Gerçek Hasta): 112', 'MIT-BIH (Gerçek Hasta): 113', 'MIT-BIH (Gerçek Hasta): 114', 'MIT-BIH (Gerçek Hasta): 115', 'MIT-BIH (Gerçek Hasta): 116', 'MIT-BIH (Gerçek Hasta): 117', 'MIT-BIH (Gerçek Hasta): 118', 'MIT-BIH (Gerçek Hasta): 119', 'MIT-BIH (Gerçek Hasta): 121', 
            'MIT-BIH (Gerçek Hasta): 122', 'MIT-BIH (Gerçek Hasta): 123', 'MIT-BIH (Gerçek Hasta): 124', 'MIT-BIH (Gerçek Hasta): 200', 'MIT-BIH (Gerçek Hasta): 201', 'MIT-BIH (Gerçek Hasta): 202', 'MIT-BIH (Gerçek Hasta): 203', 'MIT-BIH (Gerçek Hasta): 205', 'MIT-BIH (Gerçek Hasta): 207', 'MIT-BIH (Gerçek Hasta): 208', 
            'MIT-BIH (Gerçek Hasta): 209', 'MIT-BIH (Gerçek Hasta): 210', 'MIT-BIH (Gerçek Hasta): 212', 'MIT-BIH (Gerçek Hasta): 213', 'MIT-BIH (Gerçek Hasta): 214', 'MIT-BIH (Gerçek Hasta): 215', 'MIT-BIH (Gerçek Hasta): 217', 'MIT-BIH (Gerçek Hasta): 219', 'MIT-BIH (Gerçek Hasta): 220', 'MIT-BIH (Gerçek Hasta): 221',
            'PTB-XL (Gerçek Hasta): records100/00000/00001_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00002_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00003_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00004_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00005_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00006_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00007_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00008_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00009_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00010_lr', 
            'PTB-XL (Gerçek Hasta): records100/00000/00011_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00012_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00013_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00014_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00015_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00016_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00017_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00018_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00019_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00020_lr', 
            'PTB-XL (Gerçek Hasta): records100/00000/00021_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00022_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00023_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00024_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00025_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00026_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00027_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00028_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00029_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00030_lr', 
            'PTB-XL (Gerçek Hasta): records100/00000/00031_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00032_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00033_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00034_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00035_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00036_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00037_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00038_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00039_lr', 'PTB-XL (Gerçek Hasta): records100/00000/00040_lr'
        ]
        
        # --- Arayüz Kurulumu ---
        self.setup_ui()
        
        # --- Zamanlayıcılar (Timers) ---
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.run_ai_analysis)

    def setup_ai_pipeline(self):
        print("Yapay Zeka modelleri yükleniyor...")
        self.clean_pipeline = SignalCleaningPipeline(fs=self.fs)
        self.pt_detector = PanTompkinsDetector(fs=self.fs)
        self.hrv_analyzer = HRVAnalyzer()
        self.pqrst_extractor = PQRSTExtractor(fs=self.fs)
        
        self.device = get_device()
        self.model = ECGHybridModel(in_channels=1, num_classes=15).to(self.device)
        
        # Eğitilmiş gerçek model ağırlıklarını (varsa) yükle
        weights_path = os.path.join(os.path.dirname(__file__), 'ecg_model_weights_15class.pth')
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                print("✅ DERİN ÖĞRENME MODELİ (CNN) YÜKLENDİ (15 Sınıf - Gerçek AI)!")
            except Exception as e:
                print("CNN Model yükleme hatası:", e)
                
        self.model.eval()

        # XGBoost Modelini yükle
        xgb_path = os.path.join(os.path.dirname(__file__), 'xgboost_weights_15class.pkl')
        self.xgb_model = None
        if os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print("✅ MAKİNE ÖĞRENMESİ MODELİ (XGBOOST) YÜKLENDİ! (Hibrit Sistem Devrede)")
            except Exception as e:
                print("XGBoost Model yükleme hatası:", e)

    def setup_ui(self):
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#d3d3d3')
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- 1. Veri Kaynağı Paneli ---
        source_group = QGroupBox("1. Veri Kaynağı Seçimi (Data Source)")
        source_layout = QHBoxLayout()
        
        self.radio_sim = QRadioButton("Simülasyon (Hasta Seç):")
        self.radio_sim.setChecked(True)
        
        self.cmb_sim = QComboBox()
        self.cmb_sim.addItems(self.patient_records)
        
        self.rad_live = QRadioButton("Canlı Donanım (Cihaz)")
        self.radio_sd = QRadioButton("SD Karttan Oku")
        
        source_layout.addWidget(self.radio_sim, 1)
        source_layout.addWidget(self.cmb_sim, 2)
        source_layout.addWidget(self.rad_live, 1)
        source_layout.addWidget(self.radio_sd, 1)
        source_group.setLayout(source_layout)
        main_layout.addWidget(source_group)
        
        # --- 2. Kontrol Paneli ---
        control_group = QGroupBox("2. Bağlantı Kontrolleri")
        control_layout = QHBoxLayout()
        
        self.lbl_port = QLabel("COM Port:")
        self.port_combo = QComboBox()
        self.port_combo.setEnabled(False) # Başlangıçta simülasyon seçili olduğu için kapalı
        for port in PortScanner.get_available_ports():
            self.port_combo.addItem(port)
            
        self.btn_select_file = QPushButton("SD Kart (CSV/TXT) Seç")
        self.btn_select_file.setEnabled(False)
        self.btn_select_file.clicked.connect(self.select_sd_file)
        
        self.btn_start = QPushButton("BAŞLAT (START)")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 16px;")
        self.btn_start.clicked.connect(self.toggle_stream)
        
        self.lbl_info = QLabel("Sistem Hazır.")
        self.lbl_info.setStyleSheet("color: #64b5f6;")
        
        control_layout.addWidget(self.lbl_port)
        control_layout.addWidget(self.port_combo)
        control_layout.addWidget(self.btn_select_file)
        control_layout.addStretch()
        control_layout.addWidget(self.lbl_info)
        control_layout.addWidget(self.btn_start)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Radyo butonlarının etkileşimi
        self.radio_sim.toggled.connect(self.update_ui_state)
        self.rad_live.toggled.connect(self.update_ui_state)
        self.radio_sd.toggled.connect(self.update_ui_state)
        
        # --- 3. Durum (Bilgi) Paneli ---
        status_layout = QHBoxLayout()
        
        bpm_group = QGroupBox("Anlık Nabız (BPM)")
        bpm_layout = QVBoxLayout()
        self.lbl_bpm = QLabel("--")
        self.lbl_bpm.setStyleSheet("font-size: 40px; font-weight: bold; color: #ffeb3b;")
        self.lbl_bpm.setAlignment(Qt.AlignCenter)
        bpm_layout.addWidget(self.lbl_bpm)
        bpm_group.setLayout(bpm_layout)
        
        ai_group = QGroupBox("Yapay Zeka Teşhisi (1D-CNN + BiLSTM + XGBoost Hibrit)")
        ai_layout = QVBoxLayout()
        self.lbl_ai = QLabel("Veri Bekleniyor...")
        self.lbl_ai.setStyleSheet("font-size: 32px; font-weight: bold; color: #00e676;")
        self.lbl_ai.setAlignment(Qt.AlignCenter)
        
        self.lbl_summary = QLabel("Nihai Sonuç Raporu: Henüz veri yok")
        self.lbl_summary.setStyleSheet("font-size: 18px; color: #b0bec5; font-weight: bold;")
        self.lbl_summary.setAlignment(Qt.AlignCenter)
        
        ai_layout.addWidget(self.lbl_ai)
        ai_layout.addWidget(self.lbl_summary)
        ai_group.setLayout(ai_layout)
        
        status_layout.addWidget(bpm_group, 1)
        
        # --- YENİ: PQRST Paneli ---
        pqrst_group = QGroupBox("Morfoloji (PQRST)")
        pqrst_layout = QVBoxLayout()
        
        self.lbl_qrs = QLabel("QRS Genişliği: --")
        self.lbl_pr = QLabel("PR Aralığı: --")
        self.lbl_qt = QLabel("QT Süresi: --")
        
        for lbl in (self.lbl_qrs, self.lbl_pr, self.lbl_qt):
            lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #4fc3f7;")
            lbl.setAlignment(Qt.AlignCenter)
            pqrst_layout.addWidget(lbl)
            
        pqrst_group.setLayout(pqrst_layout)
        status_layout.addWidget(pqrst_group, 1)
        
        status_layout.addWidget(ai_group, 3)
        main_layout.addLayout(status_layout)
        
        # --- 4. Grafikler ---
        self.graph_layout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.graph_layout)
        
        self.p1 = self.graph_layout.addPlot(title="Ham EKG Sensör Verisi (Giyilebilir Cihazdan / Hastadan Gelen Gerçek Sinyal)")
        self.p1.enableAutoRange(axis='y')
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.curve1 = self.p1.plot(pen=pg.mkPen('#e53935', width=2))
        
        self.graph_layout.nextRow()
        
        self.p2 = self.graph_layout.addPlot(title="Yapay Zekaya Giden Filtrelenmiş Temiz Sinyal (Wavelet + Z-Score)")
        self.p2.setYRange(-4, 4)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.curve2 = self.p2.plot(pen=pg.mkPen('#1e88e5', width=2))

    def update_ui_state(self):
        """Hangi kaynak seçildiyse menüleri ona göre aç/kapat."""
        is_hw = self.rad_live.isChecked()
        is_sd = self.radio_sd.isChecked()
        
        self.port_combo.setEnabled(is_hw)
        self.lbl_port.setEnabled(is_hw)
        self.btn_select_file.setEnabled(is_sd)
        
        if is_sd and not self.sd_file_path:
            self.lbl_info.setText("Lütfen SD Kart'tan bir dosya seçin.")
        elif is_hw:
            self.lbl_info.setText("Cihazı takın ve port seçin.")
        else:
            self.lbl_info.setText("Simülatör modunda PhysioNet'ten rastgele bir hasta seçilecek.")

    def select_sd_file(self):
        """SD Kart dosyasını seçme diyaloğu."""
        file_name, _ = QFileDialog.getOpenFileName(self, "SD Kart Verisini Aç", "", "EKG Verisi (*.csv *.txt *.dat);;Tüm Dosyalar (*)")
        if file_name:
            self.sd_file_path = file_name
            self.lbl_info.setText(f"Seçilen Dosya: {os.path.basename(file_name)}")
            
    def data_callback(self, value: float):
        if not self.is_running:
            return
        self.current_time += 1.0 / self.fs
        self.raw_data_buffer.append(value)
        self.time_buffer.append(self.current_time)
        
    def toggle_stream(self):
        if not self.is_running:
            # --- BAŞLAT ---
            if self.radio_sd.isChecked() and not self.sd_file_path:
                QMessageBox.warning(self, "Hata", "Lütfen önce SD Kart içerisinden bir dosya seçin!")
                return
                
            self.is_running = True
            self.btn_start.setText("DURDUR (STOP)")
            self.btn_start.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 10px; font-size: 16px;")
            
            # Reset buffers
            self.current_time = 0.0
            self.raw_data_buffer.clear()
            self.raw_data_buffer.extend([0.0]*self.buffer_size)
            self.time_buffer.clear()
            self.time_buffer.extend(np.linspace(-self.window_seconds, 0, self.buffer_size))
            self.beat_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0} # Sayaçları sıfırla
            self.lbl_ai.setText("Analiz ediliyor...")
            self.lbl_ai.setStyleSheet("font-size: 32px; font-weight: bold; color: #9e9e9e;")
            self.lbl_summary.setText("Nihai Sonuç Raporu: Hesaplanıyor...")
            
            self.lbl_qrs.setText("QRS Genişliği: --")
            self.lbl_pr.setText("PR Aralığı: --")
            self.lbl_qt.setText("QT Süresi: --")
            
            if self.radio_sim.isChecked():
                self.start_simulator()
            elif self.rad_live.isChecked():
                self.start_hardware()
            elif self.radio_sd.isChecked():
                self.start_sd_reader()
                
            self.plot_timer.start(33) # 30 FPS
            self.ai_timer.start(2000) # Her 2 sn
        else:
            # --- DURDUR ---
            self.is_running = False
            self.btn_start.setText("BAŞLAT (START)")
            self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 16px;")
            self.plot_timer.stop()
            self.ai_timer.stop()
            
            if self.serial_reader:
                self.serial_reader.stop()
                
            # Klinik Karar Raporu Pop-Up
            total_beats = sum(self.beat_counts.values())
            if total_beats > 0:
                most_freq = max(self.beat_counts, key=self.beat_counts.get)
                PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
                MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
                ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES
                pred_name = ALL_CLASSES[most_freq]
                
                dict_15 = {
                    "NORM_MIT": ("NORMAL SİNÜS RİTMİ", "Sağlıklı Ritim. Spora ve düzenli beslenmeye devam edin."),
                    "L_MIT": ("LBBB (Sol Dal Bloğu)", "Sol dal bloğu şüphesi. Kalp yetmezliği açısından EKO istenmeli."),
                    "R_MIT": ("RBBB (Sağ Dal Bloğu)", "Sağ dal bloğu. Çoğunlukla zararsızdır ancak düzenli izlenmelidir."),
                    "V_MIT": ("PVC (Erken Karıncık Vurusu)", "Ektopik atım tespit edildi. Çarpıntı devam ederse Holter takılmalı."),
                    "A_MIT": ("APC (Erken Kulakçık Vurusu)", "Erken atım saptandı. Aşırı çay/kahve tüketimi veya strese bağlı olabilir."),
                    "NORM": ("NORMAL EKG", "Mükemmel Sağlıklı. EKG'de hiçbir problem tespit edilmedi."),
                    "IMI": ("İnferiyor MİYOKARD ENFARKTÜSÜ", "ACİL! Alt duvar kalp krizi bulgusu! Hemen acile başvurunuz."),
                    "ASMI": ("Anteroseptal MİYOKARD ENFARKTÜSÜ", "ACİL! Ön duvar kalp krizi bulgusu! Anjiyografi değerlendirilmelidir."),
                    "LVH": ("SOL KARINCIK HİPERTROFİSİ", "Kalp duvarında kalınlaşma (Kalp Büyümesi). Tansiyon takibi şarttır."),
                    "LAFB": ("Sol Ön Dal Bloğu", "İletim gecikmesi. Asemptomatik ise müdahale gerektirmeyebilir."),
                    "1AVB": ("Birinci Derece AV Blok", "Kulakçık ile karıncık arası iletim gecikmiş. İlaç veya pacemaker kontrolü."),
                    "CRBBB": ("Tam Sağ Dal Bloğu", "Sağ karıncıkta iletim blokajı. Yapısal kalp hastalığı araştırılmalı."),
                    "CLBBB": ("Tam Sol Dal Bloğu", "Sol karıncıkta iletim blokajı. Gizli iskemik kalp hastalığı riski."),
                    "AFIB": ("ATRİYAL FİBRİLASYON", "Düzensiz ve tehlikeli ritim! Pıhtı atma riski yüksek, kan sulandırıcı başlanmalı."),
                    "STACH": ("SİNÜS TAŞİKARDİSİ", "Aşırı hızlı kalp atımı (BPM > 100). Efor, ateş, hipertiroid veya stres kaynaklı olabilir.")
                }
                
                label, clinical_desc = dict_15.get(pred_name, ("BİLİNMİYOR", "Doktora danışın."))
                msg = QMessageBox(self)
                msg.setWindowTitle("Klinik Karar Raporu")
                msg.setIcon(QMessageBox.Information)
                msg.setText(f"<b>Nihai Teşhis:</b> {label}<br><br>"
                            f"<b>Klinik Karar:</b> {clinical_desc}<br><br>"
                            f"<b>İncelenen Atım:</b> {total_beats}")
                msg.exec_()
                
    def start_simulator(self):
        rec_id = self.cmb_sim.currentText()
        if "bulunamadı" in rec_id or "yok" in rec_id:
            self.lbl_info.setText("Hata: Geçerli bir dosya seçilmedi!")
            return
            
        self.current_sim_rec_id = rec_id # Demo modu için kaydet
        self.lbl_info.setText(f"Simülasyon: İnternetten {rec_id} numaralı hasta indiriliyor...")
        
        def fetch_and_start():
            try:
                if rec_id.startswith("PTB-XL"):
                    record_filename = rec_id.split(": ")[1]
                    if " (" in record_filename:
                        record_filename = record_filename.split(" ")[0]
                    
                    ptbxl_base_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
                    record_path = os.path.join(ptbxl_base_path, record_filename)
                    
                    record = wfdb.rdrecord(record_path, channels=[0])
                    ecg_mv = record.p_signal[:, 0]
                    orig_fs = record.fs
                    
                    if orig_fs != self.fs:
                        import scipy.signal
                        num_samples = int(len(ecg_mv) * self.fs / orig_fs)
                        ecg_mv = scipy.signal.resample(ecg_mv, num_samples)
                        
                    self.sim_data = ecg_mv * 1000.0
                else:
                    # İnternet üzerinden PhysioNet veri çekimi (MIT-BIH)
                    pipeline = RawDataPipeline(record_id=rec_id)
                    data = pipeline.fetch_data(duration_seconds=300)
                    self.sim_data = data["ecg_uv"]
                    
                self.sim_idx = 0
                self.beat_counts = {i: 0 for i in range(15)}
                QMetaObject.invokeMethod(self, "on_sim_data_ready", Qt.QueuedConnection)
            except Exception as e:
                print(f"Veri çekme hatası: {e}")
                
        # Arayüz donmasın diye indirme işlemini ayrı Thread'e alıyoruz
        threading.Thread(target=fetch_and_start, daemon=True).start()

    @pyqtSlot()
    def on_sim_data_ready(self):
        self.lbl_info.setText(f"Simülasyon Aktif: {self.current_sim_rec_id}")
        self._start_data_feeder_thread()

    def start_sd_reader(self):
        self.lbl_info.setText(f"SD Kart Dosyası Okunuyor: {os.path.basename(self.sd_file_path)}")
        try:
            # Dosyadan verileri oku (Her satırda bir float değer olduğu varsayımıyla)
            data = np.loadtxt(self.sd_file_path)
            self.sim_data = data / 1000.0 if np.max(np.abs(data)) > 10 else data # Basit voltaj düzeltmesi
            self.sim_idx = 0
            self.beat_counts = {i: 0 for i in range(15)}
            self._start_data_feeder_thread()
        except Exception as e:
            QMessageBox.critical(self, "Okuma Hatası", f"Dosya okunamadı: {str(e)}")
            self.toggle_stream()

    def start_hardware(self):
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "Port Hatası", "Uygun port bulunamadı.")
            self.toggle_stream()
            return
            
        self.lbl_info.setText(f"Donanım Bağlandı: {port}")
        self.serial_reader = SerialDataReader(port, baudrate=115200, callback=self.data_callback)
        self.serial_reader.start()

    def _start_data_feeder_thread(self):
        """Simülatör veya SD Kart verisini sanki canlı cihazmış gibi yavaş yavaş akıtır."""
        def feed_loop():
            start_time = time.time()
            start_idx = self.sim_idx
            while self.is_running:
                elapsed = time.time() - start_time
                expected_idx = start_idx + int(elapsed * self.fs)
                
                while self.sim_idx < expected_idx and self.is_running:
                    if self.sim_idx < len(self.sim_data):
                        self.data_callback(self.sim_data[self.sim_idx])
                        self.sim_idx += 1
                    else:
                        self.sim_idx = 0
                        start_time = time.time()
                        start_idx = 0
                        break
                time.sleep(0.01) # CPU'yu yormamak için kısa uyku
                
        self.simulator_thread = threading.Thread(target=feed_loop, daemon=True)
        self.simulator_thread.start()

    def update_plots(self):
        raw_y = np.array(self.raw_data_buffer)
        t_x = np.array(self.time_buffer)
        
        self.p1.setXRange(t_x[-1] - self.window_seconds, t_x[-1])
        self.curve1.setData(t_x, raw_y)
        
        self.p2.setXRange(t_x[-1] - self.window_seconds, t_x[-1])
        self.curve2.setData(t_x, self.clean_data_buffer)

    def run_ai_analysis(self):
        if not self.is_running: return
        raw_segment = np.array(self.raw_data_buffer)
        if np.all(raw_segment == 0): return
            
        try:
            clean_res = self.clean_pipeline.run(raw_segment * 1000.0, normalize=True)
            self.clean_data_buffer = clean_res["ecg_normalized"]
        except Exception:
            return

        r_peaks = self.pt_detector.detect(self.clean_data_buffer)
        if len(r_peaks) < 2:
            self.lbl_bpm.setText("--")
            self.lbl_ai.setText("Yeterli atım yok...")
            self.lbl_ai.setStyleSheet("font-size: 32px; font-weight: bold; color: #ff9800;")
            return
            
        rr_intervals = self.pt_detector.get_rr_intervals(r_peaks)
        bpm = self.hrv_analyzer.compute_time_domain(rr_intervals).get("mean_hr_bpm", 0)
        self.lbl_bpm.setText(f"{bpm:.0f}")
        
        # --- PQRST Analizi ---
        pqrst_list = []
        for peak in r_peaks:
            waves = self.pqrst_extractor.extract_waves(self.clean_data_buffer, peak)
            intervals = self.pqrst_extractor.compute_intervals(waves)
            pqrst_list.append(intervals)
            
        if pqrst_list:
            qrs = np.nanmean([d["QRS_Width_ms"] for d in pqrst_list])
            pr = np.nanmean([d["PR_Interval_ms"] for d in pqrst_list])
            qt = np.nanmean([d["QT_Interval_ms"] for d in pqrst_list])
            
            self.lbl_qrs.setText(f"QRS Genişliği: {qrs:.0f} ms" if not np.isnan(qrs) else "QRS Genişliği: --")
            self.lbl_pr.setText(f"PR Aralığı: {pr:.0f} ms" if not np.isnan(pr) else "PR Aralığı: --")
            self.lbl_qt.setText(f"QT Süresi: {qt:.0f} ms" if not np.isnan(qt) else "QT Süresi: --")
        
        rp = r_peaks[len(r_peaks)//2]
        start_idx = rp - int(self.fs * 0.3)
        end_idx = rp + int(self.fs * 0.5)
        
        if start_idx < 0 or end_idx > len(self.clean_data_buffer): return
        beat = self.clean_data_buffer[start_idx:end_idx]
        
        import scipy.signal
        if len(beat) != 250:
            beat = scipy.signal.resample(beat, 250)
        input_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 1. DERİN ÖĞRENME (CNN) TAHMİNİ
        with torch.no_grad():
            logits = self.model(input_tensor)
            cnn_probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            
        # 2. XGBOOST TAHMİNİ (Eğer model yüklüyse)
        xgb_probs = cnn_probs # Varsayılan olarak CNN'e eşitle
        if self.xgb_model is not None and pqrst_list:
            try:
                last_pqrst = pqrst_list[-1]
                pre_rr = rr_intervals[-1] if len(rr_intervals) > 0 else 800.0
                post_rr = pre_rr # Anlık (son atım) olduğu için
                
                # XGBoost Özellikleri: [qrs_width, pr_interval, qt_interval, pre_rr, post_rr]
                qrs_val = last_pqrst["QRS_Width_ms"] if not np.isnan(last_pqrst["QRS_Width_ms"]) else 100.0
                pr_val = last_pqrst["PR_Interval_ms"] if not np.isnan(last_pqrst["PR_Interval_ms"]) else 160.0
                qt_val = last_pqrst["QT_Interval_ms"] if not np.isnan(last_pqrst["QT_Interval_ms"]) else 400.0
                
                features = np.array([[qrs_val, pr_val, qt_val, pre_rr, post_rr]])
                xgb_probs = self.xgb_model.predict_proba(features)[0]
            except Exception as e:
                pass
                
        # 3. HİBRİT KARAR (ENSEMBLE) - CNN %60, XGBoost %40 Ağırlıklı Ortalaması
        final_probs = (cnn_probs * 0.6) + (xgb_probs * 0.4)
        
        pred_class = np.argmax(final_probs)
        confidence = final_probs[pred_class] * 100
        
        # Gerçek 15 Sınıf
        PTBXL_CLASSES = ["NORM", "IMI", "ASMI", "LVH", "LAFB", "1AVB", "CRBBB", "CLBBB", "AFIB", "STACH"]
        MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
        ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES
        
        pred_class_name = ALL_CLASSES[pred_class]
        
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
        
        label_text, color, clinical_decision = dict_15.get(pred_class_name, ("BİLİNMİYOR", "#ffffff", "Doktora danışın."))
        
        self.lbl_ai.setText(f"{label_text}\n(Güven: %{confidence:.1f})\nKlinik Karar: {clinical_decision}")
        self.lbl_ai.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")
        
        # --- GENEL SONUÇ RAPORU (NİHAİ TEŞHİS) HESAPLAMA ---
        self.beat_counts[pred_class] += 1
        total_beats = sum(self.beat_counts.values())
        # NORM_MIT ve NORM classlarını bul (0 ve 71 değil, dinamik)
        normal_idx = [i for i, c in enumerate(ALL_CLASSES) if "NORM" in c]
        normal_count = sum(self.beat_counts[i] for i in normal_idx)
        abnormal_count = total_beats - normal_count
        
        if total_beats > 0:
            risk_ratio = (abnormal_count / total_beats) * 100
            
            if risk_ratio == 0:
                risk_text = "MÜKEMMEL SAĞLIKLI"
                risk_color = "#00e676"
            elif risk_ratio < 10:
                risk_text = f"DÜŞÜK RİSK (Anormal Atım: %{risk_ratio:.1f})"
                risk_color = "#ffeb3b"
            elif risk_ratio < 30:
                risk_text = f"ORTA RİSK - DOKTORA GÖRÜNÜN (Anormal: %{risk_ratio:.1f})"
                risk_color = "#ff9800"
            else:
                risk_text = f"YÜKSEK RİSK - KRİTİK ARİTMİ (Anormal: %{risk_ratio:.1f})"
                risk_color = "#f44336"
                
            # Kayıt süresini hesapla
            elapsed = int(self.current_time)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            # Tıbbi güvenilirlik uyarısı
            warning = " (Güvenilir sonuç için en az 1 dk izleyin)" if elapsed < 60 else ""
            
            # Hastalık isimlerini topla
            detected = []
            if self.beat_counts[1] > 0: detected.append("LBBB")
            if self.beat_counts[2] > 0: detected.append("RBBB")
            if self.beat_counts[3] > 0: detected.append("PVC")
            if self.beat_counts[4] > 0: detected.append("APC")
            if self.beat_counts[5] > 0: detected.append("MI")
            if self.beat_counts[6] > 0: detected.append("İskemi(STTC)")
            if self.beat_counts[7] > 0: detected.append("İletim(CD)")
            if self.beat_counts[8] > 0: detected.append("Hipertrofi(HYP)")
            disease_str = f" -> Tespit Edilenler: {', '.join(detected)}" if detected else ""
                
            summary = f"Kayıt Süresi: {time_str}{warning}  |  İncelenen Atım: {total_beats}  |  Normal: {normal_count}  |  Anormal: {abnormal_count}\nNihai Sonuç: {risk_text}{disease_str}"
            self.lbl_summary.setText(summary)
            self.lbl_summary.setStyleSheet(f"font-size: 18px; color: {risk_color}; font-weight: bold;")


if __name__ == "__main__":
    app = QApplication(sys.path)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
