import re
import os

filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/ads1293_pipeline/module1_data_acquisition.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Update PhysioNetLoader to include retry and caching
new_fetch_record = """    def fetch_record(
        self,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        import wfdb
        import os
        import time

        # Local cache path
        cache_dir = os.path.expanduser("~/.ads1293_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self.database}_{self.record_id}_ch{self.channel}.npz")

        sampfrom = 0
        sampto = None
        
        # Load from cache if exists
        if os.path.exists(cache_file):
            print(f"[PhysioNetLoader] Önbellekten yükleniyor: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            self.fs = data["fs"].item()
            ecg_uv = data["ecg_uv"]
            raw_adc_24bit = data["raw_adc_24bit"]
            timestamps = data["timestamps"]
            annotations_dict = data["annotations"].item() if "annotations" in data and data["annotations"].item() is not None else None
            
            if annotations_dict:
                class DummyAnn:
                    def __init__(self, sample, symbol):
                        self.sample = sample
                        self.symbol = symbol
                self._annotations = DummyAnn(annotations_dict["sample"], annotations_dict["symbol"])
            else:
                self._annotations = None

            if duration_seconds is not None:
                sampto = int(duration_seconds * self.fs)
                ecg_uv = ecg_uv[:sampto]
                raw_adc_24bit = raw_adc_24bit[:sampto]
                timestamps = timestamps[:sampto]
                if self._annotations is not None:
                    mask = self._annotations.sample < sampto
                    self._annotations.sample = self._annotations.sample[mask]
                    self._annotations.symbol = list(np.array(self._annotations.symbol)[mask])

            self._raw_signal = raw_adc_24bit
            return {
                "ecg_uv": ecg_uv,
                "raw_adc_24bit": raw_adc_24bit,
                "timestamps": timestamps,
                "annotations": self._annotations,
                "fs": self.fs,
            }

        # Otherwise, fetch from PhysioNet with retry logic
        print(f"[PhysioNetLoader] PhysioNet'ten indiriliyor: {self.database}/{self.record_id}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if duration_seconds is not None:
                    header = wfdb.rdheader(self.record_id, pn_dir=self.database)
                    self.fs = header.fs
                    sampto = int(duration_seconds * self.fs)

                record = wfdb.rdrecord(
                    self.record_id,
                    pn_dir=self.database,
                    sampfrom=sampfrom,
                    sampto=sampto,
                    channels=[self.channel],
                )
                self.fs = record.fs
                ecg_mv = record.p_signal[:, 0]  # mV
                break
            except Exception as e:
                print(f"İndirme hatası (Deneme {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        else:
            raise ConnectionError(f"PhysioNet'e bağlanılamadı: {self.database}/{self.record_id}")

        try:
            self._annotations = wfdb.rdann(
                self.record_id,
                "atr",
                pn_dir=self.database,
                sampfrom=sampfrom,
                sampto=sampto,
            )
        except Exception:
            self._annotations = None

        ecg_uv = ecg_mv * 1000.0  # 1 mV = 1000 μV
        raw_adc_float = ecg_uv / self.v_lsb
        raw_adc_24bit = np.clip(
            np.round(raw_adc_float).astype(np.int32),
            -self.ADC_MAX,
            self.ADC_MAX - 1,
        )
        timestamps = np.arange(len(ecg_uv)) / self.fs
        self._raw_signal = raw_adc_24bit
        
        # Save to cache
        ann_dict = None
        if self._annotations is not None:
            ann_dict = {"sample": self._annotations.sample, "symbol": self._annotations.symbol}
            
        np.savez_compressed(
            cache_file, 
            ecg_uv=ecg_uv, 
            raw_adc_24bit=raw_adc_24bit, 
            timestamps=timestamps, 
            fs=self.fs,
            annotations=ann_dict
        )

        return {
            "ecg_uv": ecg_uv,
            "raw_adc_24bit": raw_adc_24bit,
            "timestamps": timestamps,
            "annotations": self._annotations,
            "fs": self.fs,
        }

    def get_annotation_labels(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        if self._annotations is None:
            return None
        return self._annotations.sample, self._annotations.symbol
"""

code = re.sub(r'    def fetch_record\(.*?get_annotation_labels\(.*?return self\._annotations\.sample, self\._annotations\.symbol', new_fetch_record, code, flags=re.DOTALL)

# Add fetch_data to RawDataPipeline
fetch_data_method = """    def fetch_data(self, duration_seconds: Optional[float] = None) -> Dict[str, np.ndarray]:
        \"\"\"Doğrudan PhysioNetLoader'dan veri çeker (gui_app.py uyumluluğu için).\"\"\"
        if duration_seconds is not None:
            self.duration_seconds = duration_seconds
        return self.loader.fetch_record(duration_seconds=self.duration_seconds)
"""

code = re.sub(r'    def run\(self\) -> Dict:', fetch_data_method + '\n    def run(self) -> Dict:', code)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("module1 updated successfully.")
