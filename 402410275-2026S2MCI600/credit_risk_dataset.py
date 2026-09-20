# Machine Learning 600 Assignment: Credit Risk Prediction System
# Name: ____________________     Student ITS No: ____________________
# Dataset: Credit Risk Dataset from Kaggle (credit_risk_dataset.csv)
#
# The company wants to predict whether a loan applicant will default (1) or not (0).
# This is supervised classification, because the data already has the answer
# (loan_status) and the answer is a category, not a number.
#
# To run: keep this file and the csv in the same folder. Close each graph
# window after looking at it so the code carries on.
#
# Comment labels used below:
#   Justification = why I did it
#   Interpretation = what the output tells me


# ---------------------------------------------------------------------------
# PART A: Data Acquisition and Understanding
# ---------------------------------------------------------------------------
# Load the csv with pandas and have a first look before changing anything.

import os
import pandas as pd

# Without these two lines pandas hides some columns with "..." in the output.
# The brief says all outputs must be visible, so I turned that off.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Finds the folder this file is in, so the csv still loads if the program
# is started from a different folder.
try:
    project_folder = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_folder = os.getcwd()

df = pd.read_csv(os.path.join(project_folder, 'credit_risk_dataset.csv'))

print("=" * 70)
print("PART A: DATA ACQUISITION AND UNDERSTANDING")
print("=" * 70)

# Shape is (number of rows, number of columns)
print("\nSHAPE OF DATASET (rows, columns):\n", df.shape)

print("\nCOLUMN NAMES:\n", df.columns.tolist())

# info() shows the data type of each column and how many values are not empty
print("\nDATA TYPES AND NON-NULL COUNTS:")
df.info()

print("\nSUMMARY STATS FOR NUMERIC COLUMNS:\n", df.describe())

# Same idea for the text columns (unique values, most common value, its count)
print("\nSUMMARY STATS FOR CATEGORICAL COLUMNS:\n", df.describe(exclude='number'))

print("\nMISSING VALUES PER COLUMN:\n", df.isnull().sum())

# Interpretation:
# The data has 32,581 rows and 12 columns, which is enough to train models on.
# Only two columns have missing values: person_emp_length and loan_int_rate.
# The max age is 144 and the max employment length is 123 years. Those can't
# be right, so there are data errors I need to fix in Part C.
# loan_status is the target (what I want to predict) and the other columns
# are the features (what I use to predict it).


# ---------------------------------------------------------------------------
# PART B: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
# Graphs help me spot patterns and problems that a table of numbers hides.

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 70)
print("PART B: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# Visual 1: target variable distribution
print("\nPercentage of each class in loan_status:")
print((df['loan_status'].value_counts(normalize=True) * 100).round(1))

sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution (0 = No Default, 1 = Default)')
plt.show()

# Interpretation (Visual 1):
# About 78% of the loans did not default and about 22% did, so the classes are
# imbalanced. A model that always guessed "no default" would get roughly 78%
# accuracy and still be useless. So I can't trust accuracy on its own and I
# will also look at precision, recall and F1. This is also why I use
# stratify=y when splitting the data later.

# Visual 2: correlation heatmap
# Correlation is a number from -1 to +1 that shows how two columns move
# together. Close to 0 means no real relationship.
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Interpretation (Visual 2):
# loan_percent_income (about 0.38) and loan_int_rate (about 0.34) have the
# strongest link with loan_status. So people who borrow a big part of their
# income, or who pay a high interest rate, default more often.
# person_income is slightly negative (about -0.14), so higher earners default
# a bit less.
# person_age and cb_person_cred_hist_length are very highly correlated (about
# 0.86). That makes sense because older people have had credit for longer.
# They give almost the same information, so PCA might help with this later.

# Visual 3: boxplots for outliers
# The box is the middle 50% of the values. Dots far outside the box are outliers.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df['person_age'], ax=axes[0])
sns.boxplot(y=df['person_income'], ax=axes[1])
sns.boxplot(y=df['person_emp_length'], ax=axes[2])
axes[0].set_title('Age')
axes[1].set_title('Income')
axes[2].set_title('Employment Length (years)')
plt.tight_layout()
plt.show()

# Interpretation (Visual 3):
# Age has values like 144 and employment length has 123 years. Nobody works for
# 123 years, so these are data entry errors.
# Income also has some very high values (up to 6,000,000), but a very rich
# applicant is possible, so I don't treat those as mistakes.
# I will only remove the impossible values in Part C.

# Visual 4: histograms of selected numerical features
df[['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']].hist(figsize=(10, 6), bins=30)
plt.tight_layout()
plt.show()

# Interpretation (Visual 4):
# person_age, person_income and loan_amnt are right-skewed. Most applicants
# are young with a lower income and a smaller loan, and there is a long tail
# of a few big values on the right. Most applicants are in their 20s.
# loan_int_rate looks fairly even and is centred at around 11%.
# Because of the skew I used the median (not the mean) to fill missing values.


# ---------------------------------------------------------------------------
# PART C: Data Cleaning and Preprocessing
# ---------------------------------------------------------------------------
# Fix the problems found in Parts A and B before training any model.

print("\n" + "=" * 70)
print("PART C: DATA CLEANING AND PREPROCESSING")
print("=" * 70)

# Step 1: missing values
# Justification: dropping every row with a missing value would remove about
# 4,000 rows (over 10% of the data), which is too much to lose. So I filled
# them in with the median. I chose the median over the mean because the mean
# gets pulled up by extreme values like age 144, but the median doesn't.
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
print("\nMissing values AFTER filling:\n", df.isnull().sum())

# Step 2: duplicates
# Justification: a duplicate row means the same person is counted twice. A copy
# could also end up in both the training and test set, and then the test score
# would look better than it really is.
rows_before = len(df)
df = df.drop_duplicates()
print("\nDuplicate rows removed:", rows_before - len(df))

# Step 3: outliers
# Justification: the boxplots showed ages over 100 and employment lengths over
# 60 years are mistakes, so I removed those rows. I kept the high incomes
# because they could be real customers and removing them would lose real data.
rows_before = len(df)
df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 60)]
print("Outlier rows removed:", rows_before - len(df))
print("Rows left after cleaning:", len(df))

# Step 4: encode categorical variables
# Justification: models can only work with numbers, so the text columns have to
# be converted. One-hot encoding makes a separate 0/1 column for each category.
# I did not just number them (RENT = 1, OWN = 2 and so on) because then the
# model would think OWN is "bigger" than RENT, which isn't true.
# drop_first=True drops one column for each feature. It isn't needed, because
# if all the other columns are 0 then it must be the dropped one. Keeping it
# would also make the columns copy each other, which can cause problems for
# Logistic Regression.
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print("\nShape after encoding (rows, columns):", df.shape)

# Step 5: train/test split (80/20)
# Justification: the model learns from the training set (80%). I keep the other
# 20% hidden and use it to test the model on data it has never seen, like new
# loan applicants. random_state=42 gives the same split every time I run it.
# stratify=y keeps the same default percentage (about 22%) in both sets.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set size:", X_train.shape, "| Testing set size:", X_test.shape)

# Step 6: feature scaling
# Why scaling is necessary:
# The columns are on very different scales. Income is in the tens of thousands,
# age is in the tens and the dummy columns are only 0 or 1.
# kNN works out distances between rows, so income would dominate just because
# its numbers are so big. Logistic Regression also works better when the
# features are on a similar scale. PCA needs it too, because PCA looks at
# variance and the columns with big numbers would win unfairly.
# Decision Trees don't really need scaling, but it doesn't hurt them, so I use
# the same scaled data for all three models to keep the comparison fair.
#
# I fit the scaler on the training data only and then just transform the test
# data. If I fitted it on all the data, information from the test set would
# leak into the training (this is called data leakage).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done. Example: mean of first scaled training column =",
      round(X_train_scaled[:, 0].mean(), 4))


# ---------------------------------------------------------------------------
# PART D: Principal Component Analysis (PCA)
# ---------------------------------------------------------------------------
# PCA turns the original features into a smaller set of new features called
# principal components. Each component is a mix of the original features. The
# first one holds the most information (variance), the second holds the next
# most, and so on. It is used to reduce the number of features and to remove
# overlap between features that are similar (like age and credit history length).

import numpy as np
from sklearn.decomposition import PCA

print("\n" + "=" * 70)
print("PART D: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 70)

# First keep all the components so I can see how much each one explains.
# I fit on the scaled training data only, to avoid data leakage.
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Explained variance ratio = the share of the total information in each
# component (0.10 means 10%)
print("\nExplained variance ratio (one value per component):\n", pca_full.explained_variance_ratio_)

# Cumulative variance = running total of the ratios above
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance by Number of Components')
plt.legend()
plt.show()

# Find the smallest number of components that keeps at least 95% of the
# variance. argmax gives the first position where the condition is True, and I
# add 1 because Python starts counting at 0.
# Justification: the brief says 90-95%. I chose 95% so that very little
# information is lost.
n_components = np.argmax(cum_var >= 0.95) + 1
print("Components needed for 95% variance:", n_components,
      "out of", X_train_scaled.shape[1], "original features")

# Create the PCA dataset (fit on train, then transform both train and test)
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("PCA train shape:", X_train_pca.shape, "| PCA test shape:", X_test_pca.shape)

# Interpretation:
# The first component only holds about 10% of the variance and the rest is
# spread fairly evenly, so there is no one or two components that dominate.
# It takes 17 components (out of 22) to reach 95%, so PCA only shrinks the data
# a little here.
# I think this is because most of my features, especially the 0/1 dummy
# columns, are not strongly related to each other, so there isn't much overlap
# for PCA to remove. Age and credit history length are the exception.
# Another downside is that the components are mixtures of features, so they are
# harder to explain to a manager than something like "income".


# ---------------------------------------------------------------------------
# PART E: Model Development (Before and After PCA)
# ---------------------------------------------------------------------------
# I train the three models on the original scaled features, then train the same
# three on the PCA features, and compare them.
#
# Notes on the models:
# - Logistic Regression: draws a straight line between the two classes and
#   gives a probability of default. It is simple and easy to explain.
# - k-Nearest Neighbours: looks at the k most similar past applicants (default
#   is 5) and goes with the majority. It uses distance, so scaling matters.
# - Decision Tree: asks a series of yes/no questions (like "is
#   loan_percent_income above 0.3?") until it reaches a decision. It is easy to
#   understand, but it can overfit, which means it memorises the training data.
#
# Notes on the metrics (here "positive" means 1 = default):
# - Accuracy: the share of all predictions that were correct.
# - Precision: of the people the model flagged as defaulters, how many really
#   defaulted.
# - Recall: of the people who really defaulted, how many the model caught.
# - F1-score: one number that balances precision and recall.
# - Confusion matrix: a table of right and wrong predictions.
#     TN = predicted no default and they did not default (correct)
#     TP = predicted default and they did default (correct)
#     FP = predicted default but they would have paid back. This is a false
#          rejection, so the company loses a good customer.
#     FN = predicted no default but they defaulted. This is a false approval,
#          so the company loses the loan money.
# I use F1 as the main comparison number because the classes are imbalanced.

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("\n" + "=" * 70)
print("PART E: MODEL DEVELOPMENT (BEFORE AND AFTER PCA)")
print("=" * 70)


# Gives a fresh untrained copy of the three models each time it is called,
# so the before-PCA and after-PCA models don't share anything.
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "k-Nearest Neighbours": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }


# Trains a model, tests it, prints all the required metrics and returns the scores
def evaluate_model(model, X_tr, X_te, y_tr, y_te, label):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    scores = {
        "Accuracy": accuracy_score(y_te, preds),
        "Precision": precision_score(y_te, preds),
        "Recall": recall_score(y_te, preds),
        "F1": f1_score(y_te, preds),
    }

    print(f"--- {label} ---")
    for metric_name, value in scores.items():
        print(f"{metric_name}:".ljust(11), round(value, 4))

    # Rows are what really happened, columns are what the model predicted
    cm = confusion_matrix(y_te, preds)
    cm_table = pd.DataFrame(
        cm,
        index=['Actual: No Default (0)', 'Actual: Default (1)'],
        columns=['Predicted: No Default (0)', 'Predicted: Default (1)'],
    )
    print("Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):")
    print(cm_table)
    print("Classification Report:\n", classification_report(y_te, preds))
    return scores


# Models trained on the original scaled features
print("\n=== BEFORE PCA (original scaled features) ===")
results_before = {}
for name, model in get_models().items():
    results_before[name] = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)

# Models trained on the PCA features
print(f"\n=== AFTER PCA ({n_components} principal components) ===")
results_after = {}
for name, model in get_models().items():
    results_after[name] = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test, name)

# Comparison tables
comparison_before = pd.DataFrame(results_before).T.round(4)
comparison_after = pd.DataFrame(results_after).T.round(4)
print("\nCOMPARISON TABLE - BEFORE PCA:\n", comparison_before)
print("\nCOMPARISON TABLE - AFTER PCA:\n", comparison_after)

f1_comparison = pd.DataFrame({
    "F1 Before PCA": comparison_before["F1"],
    "F1 After PCA": comparison_after["F1"],
})
f1_comparison["Change"] = (f1_comparison["F1 After PCA"] - f1_comparison["F1 Before PCA"]).round(4)
print("\nF1-SCORE BEFORE VS AFTER PCA:\n", f1_comparison)

# Interpretation (model comparison):
# PCA did not improve any of the models. F1 dropped a little for Logistic
# Regression and kNN, and dropped a lot for the Decision Tree.
# My explanation is that PCA only keeps 95% of the information, so some
# useful detail was lost. A Decision Tree also works best with the real
# features (like "income"), and PCA mixes them together so the questions the
# tree asks are less clear.
# PCA only cut the features from 22 to 17, so the gain in speed is small and
# it cost some performance. For this dataset the original features are better.
# Before PCA the Decision Tree has the best F1 and the best recall, so it
# catches the most real defaulters. kNN has the best precision, so when it says
# "default" it is usually right, but it misses more defaulters.
# Logistic Regression has the lowest recall and misses a lot of defaulters. It
# draws a straight line, so it may struggle if the real pattern is curved.
#
# Interpretation (FP and FN in a business context):
# A false approval (FN) means the company lends to someone who defaults and
# loses the money. A false rejection (FP) means it turns away a good customer
# and loses that business. Losing loan money is usually the bigger cost, so
# recall matters a lot, but if precision is too low the company rejects too
# many good customers. F1 balances both, so I choose the model with the best
# F1, which is the Decision Tree on the original features.


# ---------------------------------------------------------------------------
# PART F: Model Optimisation (Hyperparameter Tuning)
# ---------------------------------------------------------------------------
# Hyperparameters are settings that I choose before training (for example how
# deep a tree is allowed to grow). The defaults are not always the best.
# GridSearchCV tries every combination of the values I give it and scores each
# one with cross-validation. With cv=5 it splits the training data into 5
# parts, trains on 4 and checks on 1, and repeats this 5 times. That is fairer
# than trusting a single split.

from sklearn.model_selection import GridSearchCV

print("\n" + "=" * 70)
print("PART F: MODEL OPTIMISATION")
print("=" * 70)

# Select the best model: highest F1 out of all six results from Part E
all_results = {}
for name in results_before:
    all_results[name + " (Before PCA)"] = results_before[name]["F1"]
    all_results[name + " (After PCA)"] = results_after[name]["F1"]
best_model_label = max(all_results, key=all_results.get)
print("\nBest model from Part E:", best_model_label, "-> F1 =", round(all_results[best_model_label], 4))
# The best one is the Decision Tree on the original scaled features, so that is
# the one I tune.

# Settings to try for the Decision Tree:
# - max_depth: how many questions deep the tree can go (None = no limit)
# - min_samples_split: minimum rows a group needs before the tree can split it
# - min_samples_leaf: minimum rows allowed in a final group (leaf)
# - criterion: the formula used to pick the best question
# Justification: these all control how complex the tree can get, which is what
# decides whether it overfits.
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy'],
}

# scoring='f1' because of the class imbalance, cv=5 for 5-fold cross-validation,
# n_jobs=-1 uses all CPU cores so it runs faster
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV F1-score:", round(grid_search.best_score_, 4))

# Results of the tuned model on the test set
print("\n=== TUNED MODEL RESULTS ===")
best_tree = grid_search.best_estimator_
tuned_scores = evaluate_model(best_tree, X_train_scaled, X_test_scaled, y_train, y_test, "Decision Tree (Tuned)")

# Before vs after tuning
untuned_scores = results_before["Decision Tree"]
tuning_comparison = pd.DataFrame({
    "Before Tuning": untuned_scores,
    "After Tuning": tuned_scores,
})
tuning_comparison["Change"] = tuning_comparison["After Tuning"] - tuning_comparison["Before Tuning"]
print("\nDECISION TREE: BEFORE VS AFTER TUNING (test set):\n", tuning_comparison.round(4))

# Train F1 vs test F1. A big gap means the model is overfitting, so this is
# the evidence for the bias and variance explanation below.
untuned_tree = DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train)
gap_table = pd.DataFrame({
    "Train F1": [f1_score(y_train, untuned_tree.predict(X_train_scaled)),
                 f1_score(y_train, best_tree.predict(X_train_scaled))],
    "Test F1": [f1_score(y_test, untuned_tree.predict(X_test_scaled)),
                f1_score(y_test, best_tree.predict(X_test_scaled))],
}, index=["Untuned Decision Tree", "Tuned Decision Tree"])
gap_table["Gap (Train - Test)"] = gap_table["Train F1"] - gap_table["Test F1"]
print("\nTRAIN VS TEST F1 (overfitting check):\n", gap_table.round(4))
print("Tree depth - untuned:", untuned_tree.get_depth(), "| tuned:", best_tree.get_depth())

# Impact of tuning on bias and variance:
# Bias is the error you get when a model is too simple to learn the real
# pattern (underfitting). Variance is the error you get when a model is too
# sensitive to the training data, so it memorises the noise and then does worse
# on new data (overfitting).
# Before tuning, the default tree has no depth limit, so it keeps splitting
# until it fits almost every training row. The table above shows a train F1 of
# about 1.0 but a much lower test F1. That big gap means high variance
# (overfitting) and very low bias.
# After tuning, the search picked max_depth = 10 and bigger minimum sizes for
# splits and leaves. The tree can't memorise small details any more, so the
# train score drops a bit and the gap between train and test almost disappears.
# So variance went down and bias went up only slightly. The test F1 got better,
# so the trade-off was worth it.
# This is the bias-variance trade-off: a simpler model has more bias but less
# variance, and a very complex model has less bias but more variance. Tuning
# helps find a good balance in the middle.
#
# Interpretation (final results and business insight):
# Tuning improved the test F1 and accuracy, and gave a big jump in precision.
# False positives (good customers wrongly rejected) fell from about 419 to
# about 54, so the company keeps a lot more good business.
# But recall went down a bit (about 0.78 to 0.71). False negatives (defaulters
# approved by mistake) went up from about 307 to about 408, and each of those
# costs the company money. So tuning traded some recall for a lot of precision,
# and the company has to decide which mistake it is more afraid of.
# Recommendation: use the tuned Decision Tree on the original features (no
# PCA), since it has the best overall F1. A tree is also easy to explain to
# managers because the decisions follow readable rules. If missing defaulters
# is seen as worse than rejecting good customers, the model could be adjusted
# to favour recall.
# Possible next steps: try class_weight='balanced' or a lower decision
# threshold to catch more defaulters, try a Random Forest, and test on newer
# customer data before using it in real life.
