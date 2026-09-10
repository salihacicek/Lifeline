import torch
from ads1293_pipeline.module5_deep_learning import ECGHybridModel

try:
    model = ECGHybridModel(in_channels=1, num_classes=15)
    model.load_state_dict(torch.load("ecg_model_weights_15class.pth", map_location="cpu", weights_only=True))
    model.eval()
    dummy_input = torch.randn(1, 1, 250)
    logits = model(dummy_input)
    print("Model loaded successfully. Logits shape:", logits.shape)
except Exception as e:
    print("Error:", e)
