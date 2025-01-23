import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'features_and_labels.csv'  # Adjust path as needed
data = pd.read_csv(file_path)

# List all columns for debugging
print("Dataset columns:", data.columns)

# Drop unrelated columns explicitly
unrelated_columns = ['Participant','Task','mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']  # Add any other unrelated columns here
data = data.drop(columns=unrelated_columns, errors='ignore')
print(data[['low_freq_energy', 'high_freq_energy']].dtypes)
print(data['low_freq_energy'].isnull().sum())
print(data['high_freq_energy'].isnull().sum())
print(data[['low_freq_energy', 'high_freq_energy']].var())
print(data['low_freq_energy'].unique())
print(data['high_freq_energy'].unique())


# Handle missing values for numeric columns only
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
print("Numeric columns:", numeric_columns)

# Extract features and labels
X = data.drop(columns=['Label'], errors='ignore')  # Features
y = data['Label']  # Target labels

# Encode the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Check for missing numeric columns (debugging step)
print("Numeric columns in dataset:", numeric_columns)

# Balance the dataset using SMOTE (Optional)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 1. Korelasyon Matrisi
plt.figure(figsize=(10, 8))
correlation_matrix = pd.DataFrame(X_resampled, columns=X.columns).corr()  # Calculate correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# 2. Tahmin sonuçlarının karışıklık matrisi
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 3. Sınıf Bazlı Performans Görselleştirme
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
precision = [report[label]['precision'] for label in encoder.classes_]
recall = [report[label]['recall'] for label in encoder.classes_]
f1_score = [report[label]['f1-score'] for label in encoder.classes_]

x = range(len(encoder.classes_))

plt.figure(figsize=(10, 6))
plt.bar(x, precision, width=0.25, label='Precision', align='center')
plt.bar([p + 0.25 for p in x], recall, width=0.25, label='Recall', align='center')
plt.bar([p + 0.50 for p in x], f1_score, width=0.25, label='F1-Score', align='center')

plt.xticks([p + 0.25 for p in x], encoder.classes_)
plt.title("Class-wise Performance Metrics")
plt.xlabel("Classes")
plt.ylabel("Scores")
plt.legend()
plt.show()
