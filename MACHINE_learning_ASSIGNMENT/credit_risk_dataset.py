




# PART A: Data Acquisition and Understanding
import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

print("SHAPE OF DATASET:\n", df.shape)
print("\nCOLUMN NAMES:\n", df.columns.tolist())
print("\nDATA TYPES AND NON-NULL COUNTS:")
df.info()
print("\nSUMMARY STATS FOR NUMERIC COLUMNS:\n", df.describe())
print("\nSUMMARY STATS FOR CATEGORICAL COLUMNS:\n", df.describe(include='object'))
print("\nMISSING VALUES PER COLUMN:\n", df.isnull().sum())


# PART B: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Target variable distribution
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution (0 = No Default, 1 = Default)')
plt.show()
# Classes are imbalanced — far more non-defaults than defaults.

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# loan_percent_income and loan_int_rate correlate most with loan_status.

# Boxplots for outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df['person_age'], ax=axes[0])
sns.boxplot(y=df['person_income'], ax=axes[1])
sns.boxplot(y=df['person_emp_length'], ax=axes[2])
plt.tight_layout()
plt.show()
# Confirms unrealistic age (144) and employment length (123) outliers.

# Histograms of selected numerical features
df[['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']].hist(figsize=(10, 6), bins=30)
plt.tight_layout()
plt.show()
# person_age and loan_amnt are right-skewed; income has a long tail of high earners.


# PART C: Data Cleaning and Preprocessing
# Handle missing values: median imputation (robust to outliers/skew)
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

# Remove duplicates
df = df.drop_duplicates()

# Treat outliers: drop unrealistic age / employment length values
df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 60)]

# Encode categorical variables
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split features and target, then train/test (80/20)
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling (fit on train only, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Scaling matters here because income (thousands) and age (tens) sit on very
# different scales — kNN and Logistic Regression would otherwise be skewed
# toward whichever feature has the largest numbers.


# PART D: Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
import numpy as np

# Fit PCA on scaled training data
pca_full = PCA()
pca_full.fit(X_train_scaled)
print("\nExplained variance ratio:\n", pca_full.explained_variance_ratio_)

# Plot cumulative variance
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance by Number of Components')
plt.legend()
plt.show()

# Number of components needed for 95% variance
n_components = np.argmax(cum_var >= 0.95) + 1
print("Components needed for 95% variance:", n_components)

# Create PCA-transformed dataset
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("PCA train shape:", X_train_pca.shape, "| PCA test shape:", X_test_pca.shape)


# PART E: Model Development (Before and After PCA)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_tr, X_te, y_tr, y_te, label):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print(f"--- {label} ---")
    print("Accuracy: ", accuracy_score(y_te, preds))
    print("Precision:", precision_score(y_te, preds))
    print("Recall:   ", recall_score(y_te, preds))
    print("F1-score: ", f1_score(y_te, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_te, preds))
    print("Classification Report:\n", classification_report(y_te, preds))
    return f1_score(y_te, preds)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "k-Nearest Neighbours": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

print("\n=== BEFORE PCA ===")
results_before = {name: evaluate_model(m, X_train_scaled, X_test_scaled, y_train, y_test, name) for name, m in models.items()}

print("\n=== AFTER PCA ===")
results_after = {name: evaluate_model(m, X_train_pca, X_test_pca, y_train, y_test, name) for name, m in models.items()}

# Compare F1-scores before vs after PCA
comparison = pd.DataFrame({"Before PCA (F1)": results_before, "After PCA (F1)": results_after})
print("\n", comparison)

# PART F: Hyperparameter Tuning for Decision Tree
from sklearn.model_selection import GridSearchCV

# Decision Tree scored highest F1 before PCA — tune that one
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy'],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV F1-score:", grid_search.best_score_)

# Evaluate the tuned model on the test set
best_tree = grid_search.best_estimator_
tuned_f1 = evaluate_model(best_tree, X_train_scaled, X_test_scaled, y_train, y_test, "Decision Tree (Tuned)")

# Compare before vs after tuning
print(f"Decision Tree F1 — before tuning: {results_before['Decision Tree']:.4f}")
print(f"Decision Tree F1 — after tuning:  {tuned_f1:.4f}")