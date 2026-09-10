#!/bin/bash
echo "🚀 Lifeline Web Projesi Başlatılıyor..."
echo ""

# Kill any existing processes on ports 8000 and 3000
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "📡 Backend (FastAPI) başlatılıyor..."
cd "$(dirname "$0")/backend" && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

echo "🌐 Frontend (Vite) başlatılıyor..."
cd "$(dirname "$0")" && npm run dev -- --port 3000 &
FRONTEND_PID=$!

echo ""
echo "✅ Her iki sistem de başlatıldı!"
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend:  http://localhost:8000"
echo ""
echo "Tarayıcı otomatik olarak açılıyor..."
sleep 2
open "http://localhost:3000"
echo ""
echo "Durdurmak için: CTRL+C"

# Graceful shutdown
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
