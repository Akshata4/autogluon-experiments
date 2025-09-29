import pandas as pd
from autogluon.tabular import TabularPredictor

# === Config ===
MODEL_PATH = "AutogluonModels/ag-20250928_231126"   # replace with your model folder
TEST_PATH  = "data/test.csv"                         # your test dataset
OUTPUT_PATH = "data/submission.csv"                  # where to save predictions
PRED_COLUMN = "Survived"                  # new column for predictions

# Function to split Name column
def split_name_column(df):
	name_split = df['Name'].str.split(',', expand=True)
	df['Lastname'] = name_split[0].str.strip()
	title_first = name_split[1].str.strip().str.split('.', expand=True)
	df['Title'] = title_first[0].str.strip()
	df['Firstname'] = title_first[1].str.strip()
	df['Title'] = df['Title'].replace({
    'Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Lady':'Rare','Countess':'Rare','Sir':'Rare',
    'Jonkheer':'Rare','Don':'Rare','Dona':'Rare','Capt':'Rare','Col':'Rare','Major':'Rare','Rev':'Rare','Dr':'Rare'
})
	return df

# === Load model ===
predictor = TabularPredictor.load(MODEL_PATH)

# lb = predictor.leaderboard(silent=True)
# ensemble_name = lb.iloc[0]["model"]  # usually the top row is the WeightedEnsemble_* best model
# print("Using model:", ensemble_name)



# === Load test data ===
test_data = pd.read_csv(TEST_PATH)
test_data = split_name_column(test_data)

## New features
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

ticket_counts = test_data['Ticket'].value_counts()
test_data['TicketGroupSize'] = test_data['Ticket'].map(ticket_counts)

# === Make predictions ===
preds = predictor.predict(test_data)

# === Save results ===
# Add new column to test dataframe
test_data[PRED_COLUMN] = preds

# Save to CSV
test_data[['PassengerId', PRED_COLUMN]].to_csv(OUTPUT_PATH, index=False)

print(f"✅ Predictions saved to {OUTPUT_PATH} with column '{PRED_COLUMN}'")
