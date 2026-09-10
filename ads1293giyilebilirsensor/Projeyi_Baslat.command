#!/bin/bash
cd "$(dirname "$0")"

# Arkaplanda olası eski python uvicorn ve node sunucularını kapat
pkill -f "start_web_server.py"
pkill -f "vite"
lsof -ti:8085 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

echo "ADS1293 Klinik Karar Web Arayüzü Başlatılıyor..."

# Backend sunucusunu yeni terminalde aç
osascript -e 'tell app "Terminal" to do script "cd \"'$(pwd)'\" && python start_web_server.py"'

# Frontend sunucusunu yeni terminalde aç ve tarayıcıyı tetikle
osascript -e 'tell app "Terminal" to do script "cd \"'$(pwd)'/web_interface/frontend\" && npm run dev"'

# Tarayıcıyı otomatik aç
sleep 3
open http://localhost:5173
