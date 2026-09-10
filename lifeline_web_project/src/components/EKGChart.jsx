import React, { useRef, useEffect } from 'react';
import './EKGChart.css';

// EKGChart now only DRAWS - no WebSocket, no state management
const EKGChart = ({ rawData, filtData, connectionStatus }) => {
  const rawCanvasRef = useRef(null);
  const filteredCanvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const rawDataRef = useRef([]);
  const filtDataRef = useRef([]);

  // Keep refs in sync with props
  rawDataRef.current = rawData;
  filtDataRef.current = filtData;

  useEffect(() => {
    const rawCanvas = rawCanvasRef.current;
    const filtCanvas = filteredCanvasRef.current;
    if (!rawCanvas || !filtCanvas) return;

    // High-DPI canvas setup
    const setupCanvas = (canvas) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      return { ctx, w: rect.width, h: rect.height };
    };

    const rr = setupCanvas(rawCanvas);
    const ff = setupCanvas(filtCanvas);

    const drawGrid = (ctx, w, h) => {
      // Fine grid every 10px (Light Rose)
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.15)';
      ctx.lineWidth = 0.5;
      for (let x = 0; x <= w; x += 10) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += 10) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      // Bold grid every 50px (Darker Rose)
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.3)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += 50) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += 50) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      // Center line
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      ctx.setLineDash([]);
    };

    const drawSignal = (ctx, data, w, h, color) => {
      if (!data || data.length < 2) return;
      let min = Infinity, max = -Infinity;
      for (const v of data) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      let range = max - min;
      if (range < 1) range = 1;
      const yMin = min - range * 0.1;
      const yRange = (max + range * 0.1) - yMin;

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';

      const xStep = w / 599; // Fix to 600 points (MAX_POINTS - 1) so it doesn't stretch initially
      for (let i = 0; i < data.length; i++) {
        const x = i * xStep;
        const y = h - ((data[i] - yMin) / yRange) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    const render = () => {
      rr.ctx.clearRect(0, 0, rr.w, rr.h);
      ff.ctx.clearRect(0, 0, ff.w, ff.h);
      drawGrid(rr.ctx, rr.w, rr.h);
      drawGrid(ff.ctx, ff.w, ff.h);
      drawSignal(rr.ctx, rawDataRef.current, rr.w, rr.h, '#0f172a'); // Very dark slate (ink)
      drawSignal(ff.ctx, filtDataRef.current, ff.w, ff.h, '#0ea5e9'); // Sky blue
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []); // Only runs once - no WebSocket deps

  return (
    <div className="ekg-chart-container glass-panel slide-in">
      <div className="chart-header">
        <h3 className="chart-title">Canlı Osiloskop Ekranı</h3>
        <p className="chart-subtitle">
          Bağlantı Durumu:{' '}
          {connectionStatus === 'connected' && <span style={{color:'#10b981'}}>● Aktif (WebSocket Akışı)</span>}
          {connectionStatus === 'idle' && <span style={{color:'#f59e0b'}}>○ Bekleniyor...</span>}
          {connectionStatus === 'error' && <span style={{color:'#ef4444'}}>✕ Hata</span>}
        </p>
      </div>
      <div className="chart-box">
        <p className="chart-label" style={{color:'#ef4444'}}>Ham EKG Sensör Verisi (Giyilebilir Cihazdan / Hastadan Gelen Gerçek Sinyal)</p>
        <canvas ref={rawCanvasRef} className="ecg-canvas"></canvas>
      </div>
      <div className="chart-box">
        <p className="chart-label" style={{color:'#3b82f6'}}>Yapay Zekaya Giden Filtrelenmiş Temiz Sinyal (Wavelet + Z-Score)</p>
        <canvas ref={filteredCanvasRef} className="ecg-canvas"></canvas>
      </div>
    </div>
  );
};

export default EKGChart;
