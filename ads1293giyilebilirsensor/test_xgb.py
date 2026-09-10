import numpy as np
from ads1293_pipeline.module5_deep_learning import XGBoostClassifier

try:
    # Dummy data
    xgb_train_x = np.random.rand(3939, 4)
    # create labels 0 to 8
    y_train = np.random.randint(0, 9, 3939)
    
    xgb_model = XGBoostClassifier(num_classes=9, n_estimators=100)
    xgb_model.train(xgb_train_x, y_train)
    print("XGBoost trained successfully with dummy data.")
except Exception as e:
    print(f"XGBoost ERROR: {e}")
