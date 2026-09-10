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
  const [connectionStatus, setConnectionStatus] = useState('idle');
  const [showStopModal, setShowStopModal] = useState(false);
  const [llmReport, setLlmReport] = useState(null);
  const [isGeneratingLlm, setIsGeneratingLlm] = useState(false);
  const wsRef = useRef(null);
  const rawBuf = useRef([]);
  const lastStatsSent = useRef(0);
  const MAX_POINTS = 1000; // Increased buffer

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
      ws = new WebSocket(`ws://localhost:8085/ws/ecg/${encodeURIComponent(selectedFile)}`);
      wsRef.current = ws;
      
      ws.onopen = () => {
        setConnectionStatus('connected');
        setShowStopModal(false);
      };
      ws.onclose = () => setConnectionStatus('idle');
      ws.onerror = () => setConnectionStatus('error');
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          rawBuf.current.push(data.raw_point);
          if (rawBuf.current.length > MAX_POINTS) {
            rawBuf.current.shift();
          }

          // Update at ~30fps
          const now = Date.now();
          if (now - lastStatsSent.current > 33) {
            lastStatsSent.current = now;
            setRawData([...rawBuf.current]);
            setEcgStats(prev => ({ ...prev, ...data })); 
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
      setEcgStats(null);
      rawBuf.current = [];
      setIsPlaying(false);
      setShowStopModal(false);
    }
    setSelectedFile(filename);
  };

  const handlePlayToggle = (playState) => {
    setIsPlaying(playState);
    if (!playState && ecgStats?.ai) {
        setShowStopModal(true);
    } else {
        setShowStopModal(false);
    }
  };

  const generateLlmReport = async () => {
    if (!ecgStats?.ai) return;
    setIsGeneratingLlm(true);
    setLlmReport(null);
    try {
      const payload = {
        bpm: ecgStats.ai.bpm,
        qrs: ecgStats.ai.qrs,
        pr: ecgStats.ai.pr,
        qt: ecgStats.ai.qt,
        diagnosis: ecgStats.ai.diagnosis,
        confidence: ecgStats.ai.confidence,
        total_beats: ecgStats.ai.total_beats,
        abnormal_count: ecgStats.ai.abnormal_count
      };
      
      const res = await fetch('http://localhost:8085/api/generate_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      
      let i = 0;
      const text = data.report;
      setLlmReport("");
      const interval = setInterval(() => {
        setLlmReport(text.substring(0, i+1));
        i++;
        if (i >= text.length) clearInterval(interval);
      }, 30);
      
    } catch (e) {
      console.error(e);
      setLlmReport("Rapor oluşturulurken bir hata oluştu.");
    } finally {
      setIsGeneratingLlm(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header slide-in">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div>
            <h1 className="text-gradient">Klinik Karar Destek Paneli (Web V3.0)</h1>
            <p className="subtitle">15-Sınıflı AI, PQRST Ekstraksiyonu ve WebSockets</p>
          </div>
          <div style={{background: 'rgba(16, 185, 129, 0.1)', border: '1px solid #10b981', color: '#10b981', padding: '4px 12px', borderRadius: '20px', fontSize: '0.8rem'}}>
            Sistem Versiyonu: V3.0 (FastAPI Backend)
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
              <span translate="no">{ecgStats?.ai?.bpm ? Math.round(ecgStats.ai.bpm) : '--'}</span>
            </div>
          </div>
          <div className="stat-card pqrst-card">
            <h4>Morfoloji (PQRST)</h4>
            <div className="pqrst-values">
              <p>QRS Genişliği: <span translate="no">{ecgStats?.ai?.qrs ? Math.round(ecgStats.ai.qrs) : '--'}</span> ms</p>
              <p>PR Aralığı: <span translate="no">{ecgStats?.ai?.pr ? Math.round(ecgStats.ai.pr) : '--'}</span> ms</p>
              <p>QT Süresi: <span translate="no">{ecgStats?.ai?.qt ? Math.round(ecgStats.ai.qt) : '--'}</span> ms</p>
            </div>
          </div>
          <AIPanel stats={ecgStats} isPlaying={isPlaying} />
        </div>

        <div style={{display: 'flex', justifyContent: 'center', marginBottom: '1rem'}}>
          <button 
            onClick={generateLlmReport} 
            disabled={!ecgStats?.ai || isGeneratingLlm}
            style={{
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              color: 'white', padding: '12px 24px', borderRadius: '8px',
              border: 'none', cursor: (!ecgStats?.ai || isGeneratingLlm) ? 'not-allowed' : 'pointer',
              fontWeight: 'bold', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px',
              boxShadow: '0 4px 15px rgba(139, 92, 246, 0.3)', transition: 'all 0.3s ease'
            }}
          >
            {isGeneratingLlm ? '⏳ LLM Raporu Yazılıyor...' : '🤖 LLM Asistanı ile Doktor Raporu Üret'}
          </button>
        </div>
        
        {llmReport !== null && (
          <div className="slide-in" style={{
            background: '#1e293b', borderLeft: '4px solid #8b5cf6', padding: '1.5rem',
            borderRadius: '8px', marginBottom: '1.5rem', color: '#f1f5f9', fontSize: '1.15rem',
            lineHeight: '1.7', fontStyle: 'italic', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            <strong style={{color: '#8b5cf6', display: 'block', marginBottom: '10px', fontSize: '1.2rem', fontStyle: 'normal'}}>📋 LLM Klinik Değerlendirme Raporu:</strong>
            {llmReport}
            {isGeneratingLlm && <span style={{animation: 'blink 1s infinite'}}>|</span>}
          </div>
        )}

        <EKGChart
          rawData={rawData}
          filtData={rawData} // We only stream raw to avoid payload bloat, can add clean later
          connectionStatus={connectionStatus}
          heatmapData={ecgStats?.ai?.heatmap || []}
          heatmapSignal={ecgStats?.ai?.heatmap_signal || []}
        />
      </div>

      {showStopModal && ecgStats?.ai && (
        <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999
        }}>
            <div style={{
                background: '#1e293b', padding: '2rem', borderRadius: '12px', maxWidth: '600px', width: '90%',
                border: `2px solid ${ecgStats.ai.color}`, color: '#f8fafc', textAlign: 'center'
            }}>
                <h2 style={{color: ecgStats.ai.color, marginBottom: '1rem'}}>Nihai Klinik Karar Raporu</h2>
                <h3 style={{marginBottom: '1rem'}}>{ecgStats.ai.diagnosis}</h3>
                <p style={{fontSize: '1.2rem', marginBottom: '2rem', fontStyle: 'italic'}}>"{ecgStats.ai.clinical_decision}"</p>
                
                <div style={{display: 'flex', justifyContent: 'space-around', background: '#0f172a', padding: '1rem', borderRadius: '8px', marginBottom: '2rem'}}>
                    <div><b>İncelenen Atım:</b> {ecgStats.ai.total_beats}</div>
                    <div><b>Normal:</b> {ecgStats.ai.normal_count}</div>
                    <div><b>Anormal:</b> {ecgStats.ai.abnormal_count}</div>
                </div>

                <button 
                    onClick={() => setShowStopModal(false)}
                    style={{
                        background: ecgStats.ai.color, color: '#fff', border: 'none', padding: '10px 24px', 
                        fontSize: '1.1rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold'
                    }}
                >
                    Raporu Kapat
                </button>
            </div>
        </div>
      )}

    </div>
  );
};

export default Dashboard;
