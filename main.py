from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# 1. Load data (Titanic dataset as example)
# train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv")
# test_data  = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv")

# # The column we want to predict:
# label = "Survived"

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
label = 'Survived'

# 2. Create predictor and train
predictor = TabularPredictor(label=label).fit(train_data)

# 3. Make predictions
preds = predictor.predict(test_data)

# 4. Evaluate (if ground truth is available)
predictor.evaluate(test_data)
