import re
filepath = "/Users/salihacicek/Desktop/ads1293giyilebilirsensor/ads1293_pipeline/module1_data_acquisition.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

bad_str = """    def fetch_data(self, duration_seconds: Optional[float] = None) -> Dict[str, np.ndarray]:
        \"\"\"Doğrudan PhysioNetLoader'dan veri çeker (gui_app.py uyumluluğu için).\"\"\"
        if duration_seconds is not None:
            self.duration_seconds = duration_seconds
        return self.loader.fetch_record(duration_seconds=self.duration_seconds)

    def run(self) -> Dict:
        \"\"\"
        Yerel dosyayı okur ve yapay zekanın beklediği formata sokar."""
        
good_str = """    def run(self) -> Dict:
        \"\"\"
        Yerel dosyayı okur ve yapay zekanın beklediği formata sokar."""

code = code.replace(bad_str, good_str)
with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)
