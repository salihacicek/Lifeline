# -*- coding: utf-8 -*-
"""
=============================================================================
MODÜL 9: Donanım ve Seri Port (Hardware Serial) Okuyucu
=============================================================================
Bu modül, hastaya bağlanan gerçek ADS1293 donanımının bilgisayara USB
veya Bluetooth (Seri Port / UART) üzerinden gönderdiği anlık ham
EKG verilerini kesintisiz okumak için tasarlanmıştır.

1. PortScanner
   - Bilgisayardaki müsait COM/TTY portlarını bulur.
   
2. SerialDataReader
   - Ayrı bir işlem parçacığında (Thread) çalışarak donanımdan gelen
     satır bazlı veri akışını okur, sayısal (float) hale getirir ve
     verilen bir tampon belleğe (Queue veya RingBuffer) aktarır.
=============================================================================
"""

import serial
import serial.tools.list_ports
import threading
import time
import warnings
from typing import List, Callable, Optional

class PortScanner:
    """
    Sisteme bağlı olan uygun seri port cihazlarını listeler.
    """
    @staticmethod
    def get_available_ports() -> List[str]:
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

class SerialDataReader:
    """
    Donanım (Örn: Arduino, ESP32, FTDI) üzerinden gelen sürekli veri akışını
    engelleyici olmayan (non-blocking) şekilde ayrı bir Thread'de okur.
    """
    def __init__(self, port: str, baudrate: int = 115200, callback: Optional[Callable[[float], None]] = None):
        self.port = port
        self.baudrate = baudrate
        self.callback = callback # Veri okunduğunda tetiklenecek fonksiyon (örn: Queue'ya ekleme)
        
        self.serial_conn = None
        self.is_running = False
        self.thread = None

    def connect(self) -> bool:
        """Seri porta bağlantı açar."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
            # Eski verileri temizle
            self.serial_conn.reset_input_buffer()
            return True
        except serial.SerialException as e:
            warnings.warn(f"Bağlantı hatası: {e}")
            return False

    def disconnect(self):
        """Bağlantıyı güvenli şekilde kapatır."""
        self.stop()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def start(self):
        """Okuma işlemini ayrı bir thread'de başlatır."""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return
                
        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Okuma işlemini durdurur."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _read_loop(self):
        """Sürekli olarak seri porttan okuma yapan döngü (Thread içerisinde çalışır)."""
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                # Satır satır oku (ADS1293 verisi Arduino tarafından println() ile gönderiliyorsa)
                line = self.serial_conn.readline()
                if line:
                    decoded = line.decode('utf-8', errors='ignore').strip()
                    if decoded:
                        # Gelen string'i float'a çevir
                        try:
                            val = float(decoded)
                            if self.callback:
                                self.callback(val)
                        except ValueError:
                            # Parse edilemeyen hatalı satırları yoksay
                            pass
            except serial.SerialException:
                # Cihaz aniden çekildiyse vb.
                self.is_running = False
                warnings.warn("Seri port okuma sırasında bağlantı koptu!")
                break
