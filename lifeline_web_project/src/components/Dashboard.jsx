import React, { useState, useRef, useEffect } from 'react';
import PatientInfo from './PatientInfo';
import AIPanel from './AIPanel';
import EKGChart from './EKGChart';
import './Dashboard.css';

const Dashboard = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ecgStats, setEcgStats] = useState(null);
  const [rawData, setRawData] = useState([]);
  const [filtData, setFiltData] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('idle');

  const wsRef = useRef(null);
  const rawBuf = useRef([]);
  const filtBuf = useRef([]);
  const lastStatsSent = useRef(0);
  const MAX_POINTS = 600;

  useEffect(() => {
    if (!isPlaying || !selectedFile) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnectionStatus(selectedFile ? 'idle' : 'idle');
      return;
    }

    let ws;
    try {
      const startIdx = ecgStats?.idx || 0;
      const elapsed = ecgStats?.elapsed || 0;
      const analyzed = ecgStats?.analyzed_beats || 0;
      const normal = ecgStats?.normal_beats || 0;
      const abnormal = ecgStats?.abnormal_beats || 0;
      
      const query = `?start_idx=${startIdx}&start_time_offset=${elapsed}&analyzed_beats_in=${analyzed}&normal_beats_in=${normal}&abnormal_beats_in=${abnormal}`;
      ws = new WebSocket(`ws://localhost:8000/ws/ecg/${encodeURIComponent(selectedFile)}${query}`);
      wsRef.current = ws;
      
      ws.onopen = () => setConnectionStatus('connected');
      ws.onclose = () => setConnectionStatus('idle');
      ws.onerror = () => setConnectionStatus('error');
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          rawBuf.current.push(data.raw_point);
          filtBuf.current.push(data.filtered_point);
          if (rawBuf.current.length > MAX_POINTS) {
            rawBuf.current.shift();
            filtBuf.current.shift();
          }

          // Update at 30fps
          const now = Date.now();
          if (now - lastStatsSent.current > 33) {
            lastStatsSent.current = now;
            setRawData([...rawBuf.current]);
            setFiltData([...filtBuf.current]);
            setEcgStats({ ...data }); // Clone object to trigger re-render
          }
        } catch (e) { console.error("WS Parse error", e); }
      };
    } catch (e) {
      setConnectionStatus('error');
    }

    return () => {
      if (ws) ws.close();
    };
  }, [selectedFile, isPlaying]);

  const handleFileSelect = (filename) => {
    if (filename !== selectedFile) {
      setRawData([]);
      setFiltData([]);
      setEcgStats(null);
      rawBuf.current = [];
      filtBuf.current = [];
      setIsPlaying(false);
    }
    setSelectedFile(filename);
  };

  const handlePlayToggle = (playState) => {
    setIsPlaying(playState);
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header slide-in">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div>
            <h1 className="text-gradient">Klinik Karar Destek Paneli</h1>
            <p className="subtitle">Gerçek Zamanlı AI Çıkarımı (Inference) ve Ritim Holter Ekranı</p>
          </div>
          <div style={{background: 'rgba(16, 185, 129, 0.1)', border: '1px solid #10b981', color: '#10b981', padding: '4px 12px', borderRadius: '20px', fontSize: '0.8rem'}}>
            Sistem Versiyonu: V2.6 (Duraklat/Devam Et)
          </div>
        </div>
      </header>

      <div className="dashboard-content">
        <div className="controls-row">
          <PatientInfo 
            onFileSelect={handleFileSelect} 
            onPlayToggle={handlePlayToggle} 
            selectedFileExt={selectedFile}
            isPlayingExt={isPlaying}
            hasData={!!ecgStats}
          />
        </div>
        <div className="stats-row">
          <div className="stat-card bpm-card">
            <h4>Anlık Nabız (BPM)</h4>
            <div className="bpm-value">
              <span translate="no">{ecgStats?.hr ? Math.round(ecgStats.hr) : '--'}</span>
            </div>
          </div>
          <div className="stat-card pqrst-card">
            <h4>Morfoloji (PQRST)</h4>
            <div className="pqrst-values">
              <p>QRS Genişliği: <span translate="no">{ecgStats?.qrs ? Math.round(ecgStats.qrs) : '--'}</span> ms</p>
              <p>PR Aralığı: <span translate="no">{ecgStats?.pr ? Math.round(ecgStats.pr) : '--'}</span> ms</p>
              <p>QT Süresi: <span translate="no">{ecgStats?.qt ? Math.round(ecgStats.qt) : '--'}</span> ms</p>
            </div>
          </div>
          <AIPanel stats={ecgStats} isPlaying={isPlaying} />
        </div>
        <EKGChart
          rawData={rawData}
          filtData={filtData}
          connectionStatus={connectionStatus}
        />
      </div>
    </div>
  );
};

export default Dashboard;
