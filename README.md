# Lifeline EKG & Yapay Zeka Telemetri Sistemi (Klinik Karar Destek Arayüzü)

Bu depo (repository), yüksek çözünürlüklü ADS1293 giyilebilir EKG sensöründen alınan ham verilerin derin öğrenme algoritmalarıyla analiz edildiği ve hekimlere gerçek zamanlı teşhis sunan **Klinik Karar Destek Sistemini** barındırır. Proje, "Ar-Ge/Yapay Zeka Modülleri" ve "Ticari Web Arayüzü" olmak üzere iki ana klasörden (Mono-Repo) oluşmaktadır.

---

## 1. Verileri Nereden Aldık ve Nasıl İşledik?

Projede kullanılan EKG verileri, sentetik veya uydurma veriler değil; doğrudan tıp dünyasının küresel altın standardı kabul edilen **PhysioNet** platformundan çekilmiştir:

*   **MIT-BIH Arrhythmia Database:** Farklı aritmi türlerini barındıran ve kardiyologlar tarafından işaretlenmiş (anotasyonlu) klinik EKG kayıtlarıdır. Derin öğrenme modelimizin temel eğitiminde kullanılmıştır.
*   **Kaggle Özelleştirilmiş Veri Setleri:** Özellikle "Kalp Krizi (Miyokard Enfarktüsü - MI)" ve "Atriyal Fibrilasyon (AFIB)" gibi hayati tehlike taşıyan kritik hastalıkların yapay zeka tarafından daha iyi öğrenilebilmesi için Kaggle'dan elde edilen özel hastalık veri setleri model eğitimine (hibrit mimari) dahil edilmiştir.
*   **PTB-XL Database:** Binlerce hastadan alınmış devasa 10 saniyelik EKG kayıtlarıdır. Algoritmanın klinik ortam genelleştirme yeteneğini test etmek için kullanılmıştır.
*   **Donanım Simülasyonu (ADS1293):** PhysioNet ve Kaggle'dan çekilen veriler sisteme doğrudan sokulmamış, öncelikle yazılımsal bir ADC filtresinden geçirilmiştir. Veriler **V_REF=2.4V** ve **PGA Gain=3.5x** değerleriyle 24-bit Two's Complement formatına çevrilerek donanımsal bir test ortamı yaratılmıştır.

---

## 2. Neyi, Hangi Teknolojilerle (Nasıl) Yaptık?

### A. Arka Plan Mimarisi (Backend & Sinyal İşleme)
*   **Teknolojiler:** `Python`, `FastAPI`, `WebSocket`, `SciPy`, `NumPy`
*   **Nasıl Yaptık?** Cihazdan (veya veritabanından) gelen EKG verileri, asenkron `FastAPI` sunucumuz üzerinden `WebSocket` ile saniyede 360 paket (360Hz) hızında ön yüze fırlatılmaktadır. Sinyal işlemede baseline wander (taban çizgisi kayması) ve parazitleri temizlemek için Bandpass filtreleri ile **Pan-Tompkins** algoritması kullanılarak milisaniyelik QRS, PR, QT genişlikleri hesaplanmıştır.

### B. Yapay Zeka ve Karar Destek Motoru (AI Engine)
*   **Teknolojiler:** `PyTorch`, `XGBoost`
*   **Nasıl Yaptık?** Basit eşik (threshold) değerleri yerine, uzamsal ve zamansal sinyal özelliklerini (Spatio-Temporal) aynı anda algılayabilen karmaşık bir hibrit model geliştirdik.
    *   **1D-CNN (Çoklu Ölçekli):** Dalgadaki milisaniyelik değişimleri yakalar.
    *   **BiLSTM (Çift Yönlü Hafıza):** Atımların geçmişini ve geleceğini analiz ederek ritim düzensizliğini anlar.
    *   **Bahdanau Attention:** Sistemin, uzun EKG dalgasındaki en şüpheli ve kritik "Anormal" bölgelere odaklanmasını sağlar.

### C. Gelişmiş Modüller: XAI ve LLM
*   **Açıklanabilir Yapay Zeka (Grad-CAM):** Doktorların yapay zekaya güvenmesi için siyah kutu (Blackbox) problemini çözdük. Sistem "Kalp Krizi" teşhisi koyduğunda, dalganın tam olarak hangi milisaniyesine bakarak bu kararı verdiğini **Isı Haritası (Heatmap)** ile ekranda renkli olarak gösterir.
*   **Üretken Yapay Zeka (LLM Raporlama):** Yapay zeka ve sinyal analizinden elde edilen sonuçları (BPM, PR mesafesi, güven skoru), hekimin saniyeler içinde okuyabilmesi için ChatGPT benzeri doğal dilde (Türkçe) otomatik bir tıbbi ön-rapora dönüştürür.

### D. Ön Yüz Arayüzü (Frontend)
*   **Teknolojiler:** `React`, `Vite`, `Tailwind CSS`, `Recharts`
*   **Nasıl Yaptık?** Gelen WebSocket verilerini anlık olarak yakalayan ve hastanelerdeki "Osiloskop" ekranlarını taklit eden, saniyede 60 kare (60 FPS) hızında akan pürüzsüz bir EKG grafiği kodlandı. Sağ bölmede ise yapay zeka sonuçları ve hasta durumları canlı olarak güncellenmektedir.

---

## 3. KVKK ve Veri Güvenliği Standardı
Geliştirilen bu sistem tamamen **Anonimleştirme (Veri Maskeleme)** prensibiyle çalışmaktadır. Sistem EKG sinyalini işlerken hastanın Ad, Soyad veya TC Kimlik numarasına ihtiyaç duymaz, verileri rastgele atanan `Hasta_ID`'ler üzerinden işler. Gelecek (Ticarileşme) vizyonunda, hastane verilerinin dışarı çıkmasını önlemek adına **Federated Learning (Birleştirilmiş Öğrenme)** mimarisinin entegrasyonu planlanmıştır.
