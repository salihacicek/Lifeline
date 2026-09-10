import React from 'react';
import './AIPanel.css';

const AIPanel = ({ stats, isPlaying }) => {
  // Use '--' if missing
  const fmt = (val) => (val !== null && val !== undefined) ? String(val) : '--';

  const ai = stats?.ai;
  const total_beats = ai?.total_beats || 0;
  const normal_count = ai?.normal_count || 0;
  const abnormal_count = ai?.abnormal_count || 0;
  const confidence = ai?.confidence !== undefined ? ai.confidence.toFixed(1) : '--';
  const record_time = stats?.record_time || '--';

  let ai_diagnosis = ai?.diagnosis || 'SİNYAL BEKLENİYOR...';
  let diagColor = ai?.color || '#94a3b8';
  let clinical_comment = ai?.clinical_decision || 'Sinyal analizi bekleniyor...';

  let risk_text = "BEKLENİYOR...";
  let risk_color = "#f59e0b"; // orange default

  if (total_beats > 0) {
    const risk_ratio = (abnormal_count / total_beats) * 100;
    if (risk_ratio === 0) {
      risk_text = "MÜKEMMEL SAĞLIKLI";
      risk_color = "#10b981"; // green
    } else if (risk_ratio < 10) {
      risk_text = `DÜŞÜK RİSK (Anormal Atım: %${risk_ratio.toFixed(1)})`;
      risk_color = "#eab308"; // yellow
    } else if (risk_ratio < 30) {
      risk_text = `ORTA RİSK - DOKTORA GÖRÜNÜN (Anormal: %${risk_ratio.toFixed(1)})`;
      risk_color = "#f59e0b"; // orange
    } else {
      risk_text = `YÜKSEK RİSK - KRİTİK ARİTMİ (Anormal: %${risk_ratio.toFixed(1)})`;
      risk_color = "#ef4444"; // red
    }
  }

  const showComment = total_beats > 0 && !isPlaying;

  return (
    <div className="stat-card ai-panel">
      <h4>Yapay Zeka Teşhisi (15 Sınıflı Hibrit AI)</h4>
      
      <div className="ai-content">
        <h2 style={{color: diagColor, fontSize: '1.5rem', textAlign: 'center', marginBottom: '8px'}}><span translate="no">{ai_diagnosis}</span></h2>
        <h3 style={{color: diagColor, fontSize: '1.2rem', textAlign: 'center', marginBottom: '16px'}}>(AI Güveni: %<span translate="no">{confidence}</span>)</h3>
        
        <p style={{color: '#f97316', textAlign: 'center', fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '8px'}}>
          Kayıt Süresi: <span translate="no">{record_time}</span> | Atım: <span translate="no">{total_beats}</span> | Normal: <span translate="no">{normal_count}</span> | Anormal: <span translate="no">{abnormal_count}</span>
        </p>
        <p style={{color: '#f97316', textAlign: 'center', fontSize: '1.1rem', fontWeight: 'bold'}}>
          Nihai Durum: <span style={{color: risk_color}} translate="no">{risk_text}</span>
        </p>
        
        {showComment && (
          <div style={{marginTop: '16px', padding: '12px', width: '100%', background: 'rgba(0,0,0,0.03)', borderRadius: '8px', borderLeft: `4px solid ${diagColor}`}}>
            <p style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 'bold'}}>Klinik Yorum (Doktor Tavsiyesi)</p>
            <p style={{fontSize: '1rem', color: diagColor, fontWeight: '500', margin: 0, fontStyle: 'italic'}} translate="no">"{clinical_comment}"</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIPanel;
