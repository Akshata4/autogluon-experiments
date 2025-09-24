import pandas as pd
from autogluon.tabular import TabularPredictor

# === Config ===
MODEL_PATH = "AutogluonModels/ag-20250924_191554"   # replace with your model folder
TEST_PATH  = "data/test.csv"                         # your test dataset
OUTPUT_PATH = "data/submission.csv"                  # where to save predictions
PRED_COLUMN = "PredictedLabel"                  # new column for predictions

# === Load model ===
predictor = TabularPredictor.load(MODEL_PATH)

# lb = predictor.leaderboard(silent=True)
# ensemble_name = lb.iloc[0]["model"]  # usually the top row is the WeightedEnsemble_* best model
# print("Using model:", ensemble_name)

# === Load test data ===
test_df = pd.read_csv(TEST_PATH)

# === Make predictions ===
preds = predictor.predict(test_df)

# === Save results ===
# Add new column to test dataframe
test_df[PRED_COLUMN] = preds

# Save to CSV
test_df[['PassengerId', PRED_COLUMN]].to_csv(OUTPUT_PATH, index=False)

print(f"✅ Predictions saved to {OUTPUT_PATH} with column '{PRED_COLUMN}'")
