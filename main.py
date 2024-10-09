import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

customers_df = pd.read_csv(r'D:\SkyHack\assets\customers2afd6ea.csv')
reason_df = pd.read_csv(r'D:\SkyHack\assets\reason18315ff.csv')
sentiment_df = pd.read_csv(r'D:\SkyHack\assets\sentiment_statisticscc1e57a.csv')
calls_df = pd.read_csv(r'D:\SkyHack\assets\callsf0d4f5a.csv')
test_df = pd.read_csv(r'D:\SkyHack\assets\testbc7185d.csv')

merged_df = calls_df.merge(customers_df, on='customer_id', how='left')
merged_df = merged_df.merge(reason_df, on='call_id', how='left')
merged_df = merged_df.merge(sentiment_df, on='call_id', how='left')

merged_df['call_start_datetime'] = pd.to_datetime(merged_df['call_start_datetime'])
merged_df['agent_assigned_datetime'] = pd.to_datetime(merged_df['agent_assigned_datetime'])
merged_df['call_end_datetime'] = pd.to_datetime(merged_df['call_end_datetime'])

# Calculate Average Handling Time (AHT) and Average Silent Time (AST)
merged_df['AHT'] = (merged_df['call_end_datetime'] - merged_df['agent_assigned_datetime']).dt.total_seconds()
merged_df['AST'] = (merged_df['agent_assigned_datetime'] - merged_df['call_start_datetime']).dt.total_seconds()

merged_df = merged_df[(merged_df['AHT'] > 0) & (merged_df['AST'] > 0)]

# Handle NaN values in silence_percent_average
merged_df['silence_percent_average'] = merged_df['silence_percent_average'].fillna(0)

# Exploratory Data Analysis (EDA)
# Visualize the distribution of AHT and AST
plt.figure(figsize=(12, 6))
sns.histplot(merged_df['AHT'], bins=50, kde=True)
plt.title("Distribution of AHT")
plt.xlabel("AHT (seconds)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(merged_df['AST'], bins=50, kde=True)
plt.title("Distribution of AST")
plt.xlabel("AST (seconds)")
plt.ylabel("Frequency")
plt.show()

# Analyzing Call Reasons
call_reason_counts = merged_df['primary_call_reason'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=call_reason_counts.index, y=call_reason_counts.values)
plt.title("Most Common Call Reasons")
plt.xlabel("Call Reason")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

# Analyze Sentiment and AHT
plt.figure(figsize=(12, 6))
sns.boxplot(x='average_sentiment', y='AHT', data=merged_df)
plt.title("AHT by Average Sentiment")
plt.show()

# Analyzing agent and customer tones
plt.figure(figsize=(12, 6))
sns.boxplot(x='agent_tone', y='AHT', data=merged_df)
plt.title("AHT by Agent Tone")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='customer_tone', y='AHT', data=merged_df)
plt.title("AHT by Customer Tone")
plt.show()

# Feature Engineering for Prediction
# Handle categorical features (encode labels)
label_encoders = {}
for col in ['agent_tone', 'customer_tone', 'primary_call_reason']:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))
    label_encoders[col] = le

# Feature selection
features = ['AHT', 'AST', 'silence_percent_average', 'agent_tone', 'customer_tone', 'average_sentiment']
X = merged_df[features]
y = merged_df['primary_call_reason']

# Check for NaN values in the feature set
if X.isnull().values.any() or y.isnull().values.any():
    print("There are NaN values in the features or target. Please check your data.")
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Optional: Test Predictions
    # Prepare test features for prediction
    test_df['call_start_datetime'] = pd.to_datetime(test_df['call_start_datetime'])
    test_df['agent_assigned_datetime'] = pd.to_datetime(test_df['agent_assigned_datetime'])
    test_df['call_end_datetime'] = pd.to_datetime(test_df['call_end_datetime'])

    # Calculate AHT and AST for test data
    test_df['AHT'] = (test_df['call_end_datetime'] - test_df['agent_assigned_datetime']).dt.total_seconds()
    test_df['AST'] = (test_df['agent_assigned_datetime'] - test_df['call_start_datetime']).dt.total_seconds()

    # Handle NaN values in silence_percent_average for test data
    test_df['silence_percent_average'] = test_df['silence_percent_average'].fillna(0)

    # Feature selection for test data
    test_features = test_df[features]

    # Check for NaN values in the test features
    if test_features.isnull().values.any():
        print("There are NaN values in the test features. Please check your data.")
    else:
        # Make predictions on test data
        test_predictions = model.predict(test_features)
