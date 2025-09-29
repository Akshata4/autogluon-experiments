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

# Split and save train
train_data = split_name_column(train_data)

## New features
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

ticket_counts = train_data['Ticket'].value_counts()
train_data['TicketGroupSize'] = train_data['Ticket'].map(ticket_counts)



# Split and save test
test_data = split_name_column(test_data)

# 2. Create predictor and train
predictor = TabularPredictor(label=label, problem_type='binary', eval_metric='accuracy').fit(train_data=train_data,
    presets='best_quality',                 # stronger than 'medium_quality'
    time_limit=1800,                        # e.g., 30 minutes
    num_bag_folds=5,                        # internal CV bagging
    num_bag_sets=1,
    num_stack_levels=1,                     # try 1 → 2 if you have time
    ag_args_fit={'verbosity': 1})
# predictor = TabularPredictor(label=label).fit(train_data)

# 3. Make predictions
preds = predictor.predict(test_data)

# 4. Evaluate (if ground truth is available)
# predictor.evaluate(test_data)
