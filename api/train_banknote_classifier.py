import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load embeddings
df = pd.read_feather('banknote_net_assets/banknote_net.feather')
# print(df.columns)
# exit()

# Filter for your target currencies (e.g., USD, GBP, EUR)
target_currencies = ['USD', 'GBP', 'EUR']
df = df[df['Currency'].isin(target_currencies)]

# Features and labels
X = df.iloc[:, :256].values  # Embeddings
y = df['Currency'] + '_' + df['Denomination'].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save classifier
joblib.dump(clf, 'banknote_net_assets/banknote_classifier.joblib')

# Print accuracy
print('Test accuracy:', clf.score(X_test, y_test))