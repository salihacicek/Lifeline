import React, { useState, useEffect } from 'react';
import { Database, CheckCircle, AlertCircle } from 'lucide-react';
import './PatientInfo.css';

const PatientInfo = ({ onFileSelect, onPlayToggle, selectedFileExt, isPlayingExt, hasData }) => {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('idle'); // idle, error

  useEffect(() => {
    // Fetch available files from backend on load
    const fetchFiles = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/files');
        const data = await response.json();
        if (data.status === 'success') {
          setFiles(data.files);
          if (data.files.length > 0 && !selectedFileExt) {
            onFileSelect(data.files[0]);
          }
        } else {
          setStatus('error');
        }
      } catch (err) {
        console.error("Error fetching files:", err);
        setStatus('error');
      }
    };
    fetchFiles();
  }, []);

  const handleStartAnalysis = () => {
    if (!selectedFileExt) return;
    onPlayToggle(true);
  };

  const handleStopAnalysis = () => {
    onPlayToggle(false);
  };

  return (
    <div className="patient-info-container glass-panel slide-in">
      <div className="upload-header">
        <h2>Veri Kaynağı Seçimi</h2>
      </div>
      
      <div className="upload-area">
        <div className="dropdown-container">
            <Database size={20} className="dropdown-icon" />
            <select 
              className="file-dropdown" 
              value={selectedFileExt || ''} 
              onChange={(e) => {
                  onFileSelect(e.target.value);
              }}
            >
              {files.length === 0 && <option value="">Dosya bulunamadı...</option>}
              {files.map((file, idx) => (
                <option key={idx} value={file}>{file}</option>
              ))}
            </select>
        </div>
        
        <button 
          className={`upload-button ${isPlayingExt ? 'success' : 'idle'}`}
          onClick={isPlayingExt ? handleStopAnalysis : handleStartAnalysis}
          disabled={!selectedFileExt}
        >
          {!isPlayingExt && <span>{hasData ? "Devam Et" : "Analizi Başlat"}</span>}
          {isPlayingExt && <span style={{color: '#fff', fontWeight: 'bold'}}>DURAKLAT (PAUSE)</span>}
          {status === 'error' && <span><AlertCircle size={16} style={{marginRight: '8px'}}/> Hata</span>}
        </button>
      </div>
    </div>
  );
};

export default PatientInfo;
