import React from 'react';
import './AIPanel.css';

const AIPanel = ({ stats, isPlaying }) => {
  // Use '--' if missing
  const fmt = (val) => (val !== null && val !== undefined) ? String(val) : '--';

  const total_beats = stats?.analyzed_beats || 0;
  const normal_count = stats?.normal_beats || 0;
  const abnormal_count = stats?.abnormal_beats || 0;
  const confidence = stats?.ai_confidence !== undefined ? stats.ai_confidence.toFixed(1) : '--';
  const record_time = stats?.record_time || '--';
  const elapsed = stats?.elapsed || 0;

  let ai_diagnosis = stats?.ai_diagnosis || 'SİNYAL BEKLENİYOR...';
  // Let's decide colors based on diagnosis string
  let isNormal = ai_diagnosis.includes("NORMAL") || ai_diagnosis.includes("Sağlıklı");
  let diagColor = isNormal ? '#10b981' : (ai_diagnosis.includes("BEKLEN") ? '#94a3b8' : '#ef4444');

  let risk_text = "BEKLENİYOR...";
  let risk_color = "#f59e0b"; // orange default
  let disease_str = "";

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
    
    // As per the old app logic, if abnormal beats > 0, it means LBBB/RBBB etc.
    if (abnormal_count > 0) {
      disease_str = " -> Tespit Edilenler: LBBB";
    }
  }

  const warning = elapsed < 60 ? " (Güvenilir sonuç için en az 1 dk izleyin)" : "";

  let clinical_comment = "Sinyal analizi bekleniyor...";
  let comment_color = "#94a3b8";

  if (total_beats > 0) {
    const risk_ratio = (abnormal_count / total_beats) * 100;
    
    if (risk_ratio === 0) {
      clinical_comment = "Hastanın ritim bulguları olağanüstü düzeydedir ve kalp kası elektriksel aktivitesi tamamen normaldir. Düzenli kardiyo egzersizleri (tempolu yürüyüş, yüzme vb.) ve dengeli beslenme ile bu sağlıklı ritmin korunması tavsiye edilir. Herhangi bir tıbbi müdahaleye gerek yoktur.";
      comment_color = "#059669"; // Emerald 600
    } else if (risk_ratio < 10) {
      clinical_comment = "Kardiyak ritimde çok nadir ve tehlikesiz erken (prematür) atımlar gözlemlenmiştir. Genellikle aşırı stres, uykusuzluk veya fazla kafein tüketimi kaynaklı olabilmektedir. Yaşam tarzı değişiklikleri (kafein azaltımı, uyku düzeni) önerilir. Şu aşamada ilaç tedavisine gerek duyulmamaktadır.";
      comment_color = "#d97706"; // Amber 600
    } else if (risk_ratio < 30) {
      clinical_comment = "DİKKAT: Orta düzeyde ritim düzensizliği (Taşikardi/Aritmi) saptanmıştır. Bu durum çarpıntı ve halsizliğe yol açabilir. Farmakolojik (ilaç) tedaviye başlanıp başlanmayacağının değerlendirilmesi için en kısa sürede Uzman Kardiyolog tarafından detaylı muayene (Holter/EKO) yapılması önerilir.";
      comment_color = "#ea580c"; // Orange 600
    } else {
      clinical_comment = "KRİTİK BULGU: Yüksek oranda tehlikeli ritim bozukluğu saptanmıştır! Hastanın derhal kardiyoloji acil servisine sevki, damar yoluyla (IV) acil ritim düzenleyici ilaç müdahalesi ve gerekirse ablasyon (yakma) veya kalp pili (Pacemaker) ameliyatı açısından acilen değerlendirilmesi hayati önem taşır.";
      comment_color = "#dc2626"; // Red 600
    }
  }

  const showComment = total_beats > 0 && !isPlaying;

  return (
    <div className="stat-card ai-panel">
      <h4>Yapay Zeka Teşhisi (1D-CNN + BiLSTM + XGBoost Hibrit)</h4>
      
      <div className="ai-content">
        <h2 style={{color: diagColor, fontSize: '1.75rem', textAlign: 'center', marginBottom: '8px'}}><span translate="no">{ai_diagnosis}</span></h2>
        <h3 style={{color: diagColor, fontSize: '1.5rem', textAlign: 'center', marginBottom: '16px'}}>(Hibrit Yapay Zeka Güveni: %<span translate="no">{confidence}</span>)</h3>
        
        <p style={{color: '#f97316', textAlign: 'center', fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '8px'}}>
          Kayıt Süresi: <span translate="no">{record_time}</span>{warning} | İncelenen Atım: <span translate="no">{total_beats}</span> | Normal: <span translate="no">{normal_count}</span> | Anormal: <span translate="no">{abnormal_count}</span>
        </p>
        <p style={{color: '#f97316', textAlign: 'center', fontSize: '1.1rem', fontWeight: 'bold'}}>
          Nihai Sonuç: <span style={{color: risk_color}} translate="no">{risk_text}</span><span translate="no">{disease_str}</span>
        </p>
        
        {showComment && (
          <div style={{marginTop: '16px', padding: '12px', width: '100%', background: 'rgba(0,0,0,0.03)', borderRadius: '8px', borderLeft: `4px solid ${comment_color}`}}>
            <p style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 'bold'}}>Klinik Yorum (Yapay Zeka Karar Destek Önerisi)</p>
            <p style={{fontSize: '1rem', color: comment_color, fontWeight: '500', margin: 0, fontStyle: 'italic'}} translate="no">"{clinical_comment}"</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIPanel;
