import torch
import numpy as np
from ads1293_pipeline.module5_deep_learning import ECGHybridModel
import pandas as pd

scp_df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv", index_col=0)
PTBXL_CLASSES = list(scp_df.index)
MIT_CLASSES = ["NORM_MIT", "L_MIT", "R_MIT", "V_MIT", "A_MIT"]
ALL_CLASSES = PTBXL_CLASSES + MIT_CLASSES

model = ECGHybridModel(in_channels=1, num_classes=76)
model.load_state_dict(torch.load("ecg_model_weights_76class.pth", map_location="cpu", weights_only=True))
model.eval()

# Dummy input
dummy_input = torch.randn(10, 1, 250)
logits = model(dummy_input)
probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
preds = np.argmax(probs, axis=1)

print("Predictions for random noise:", preds)
for p in preds:
    print(ALL_CLASSES[p], scp_df.loc[ALL_CLASSES[p]]['diagnostic_class'] if ALL_CLASSES[p] in PTBXL_CLASSES else "MIT")
