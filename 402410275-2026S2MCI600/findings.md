======================================================================
PART A: DATA ACQUISITION AND UNDERSTANDING
======================================================================

SHAPE OF DATASET (rows, columns):
 (32581, 12)

COLUMN NAMES:
 ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_status', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']

DATA TYPES AND NON-NULL COUNTS:
<class 'pandas.DataFrame'>
RangeIndex: 32581 entries, 0 to 32580
Data columns (total 12 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   person_age                  32581 non-null  int64  
 1   person_income               32581 non-null  int64  
 2   person_home_ownership       32581 non-null  str    
 3   person_emp_length           31686 non-null  float64
 4   loan_intent                 32581 non-null  str    
 5   loan_grade                  32581 non-null  str    
 6   loan_amnt                   32581 non-null  int64  
 7   loan_int_rate               29465 non-null  float64
 8   loan_status                 32581 non-null  int64  
 9   loan_percent_income         32581 non-null  float64
 10  cb_person_default_on_file   32581 non-null  str    
 11  cb_person_cred_hist_length  32581 non-null  int64  
dtypes: float64(3), int64(5), str(4)
memory usage: 3.0 MB

SUMMARY STATS FOR NUMERIC COLUMNS:
          person_age  person_income  person_emp_length     loan_amnt  loan_int_rate   loan_status  loan_percent_income  cb_person_cred_hist_length
count  32581.000000   3.258100e+04       31686.000000  32581.000000   29465.000000  32581.000000         32581.000000                32581.000000
mean      27.734600   6.607485e+04           4.789686   9589.371106      11.011695      0.218164             0.170203                    5.804211
std        6.348078   6.198312e+04           4.142630   6322.086646       3.240459      0.413006             0.106782                    4.055001
min       20.000000   4.000000e+03           0.000000    500.000000       5.420000      0.000000             0.000000                    2.000000
25%       23.000000   3.850000e+04           2.000000   5000.000000       7.900000      0.000000             0.090000                    3.000000
50%       26.000000   5.500000e+04           4.000000   8000.000000      10.990000      0.000000             0.150000                    4.000000
75%       30.000000   7.920000e+04           7.000000  12200.000000      13.470000      0.000000             0.230000                    8.000000
max      144.000000   6.000000e+06         123.000000  35000.000000      23.220000      1.000000             0.830000                   30.000000

SUMMARY STATS FOR CATEGORICAL COLUMNS:
        person_home_ownership loan_intent loan_grade cb_person_default_on_file
count                  32581       32581      32581                     32581
unique                     4           6          7                         2
top                     RENT   EDUCATION          A                         N
freq                   16446        6453      10777                     26836

MISSING VALUES PER COLUMN:
 person_age                       0
person_income                    0
person_home_ownership            0
person_emp_length              895
loan_intent                      0
loan_grade                       0
loan_amnt                        0
loan_int_rate                 3116
loan_status                      0
loan_percent_income              0
cb_person_default_on_file        0
cb_person_cred_hist_length       0
dtype: int64

======================================================================
PART B: EXPLORATORY DATA ANALYSIS (EDA)
======================================================================

Percentage of each class in loan_status:
loan_status
0    78.2
1    21.8
Name: proportion, dtype: float64

======================================================================
PART C: DATA CLEANING AND PREPROCESSING
======================================================================

Missing values AFTER filling:
 person_age                    0
person_income                 0
person_home_ownership         0
person_emp_length             0
loan_intent                   0
loan_grade                    0
loan_amnt                     0
loan_int_rate                 0
loan_status                   0
loan_percent_income           0
cb_person_default_on_file     0
cb_person_cred_hist_length    0
dtype: int64

Duplicate rows removed: 165
Outlier rows removed: 7
Rows left after cleaning: 32409

Shape after encoding (rows, columns): (32409, 23)
Training set size: (25927, 22) | Testing set size: (6482, 22)
Scaling done. Example: mean of first scaled training column = 0.0

======================================================================
PART D: PRINCIPAL COMPONENT ANALYSIS (PCA)
======================================================================

Explained variance ratio (one value per component):
 [0.10354544 0.09477754 0.07509467 0.06791914 0.05726364 0.05547288
 0.0545809  0.05327442 0.05124557 0.05027907 0.04753548 0.04615099
 0.04602243 0.04513102 0.04105337 0.03827369 0.02594388 0.02104318
 0.00879051 0.00842199 0.00551811 0.00266209]
Components needed for 95% variance: 17 out of 22 original features
PCA train shape: (25927, 17) | PCA test shape: (6482, 17)

======================================================================
PART E: MODEL DEVELOPMENT (BEFORE AND AFTER PCA)
======================================================================

=== BEFORE PCA (original scaled features) ===
--- Logistic Regression ---
Accuracy:   0.8645
Precision:  0.7581
Recall:     0.5592
F1:         0.6437
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4811                     253
Actual: Default (1)                           625                     793
Classification Report:
               precision    recall  f1-score   support

           0       0.89      0.95      0.92      5064
           1       0.76      0.56      0.64      1418

    accuracy                           0.86      6482
   macro avg       0.82      0.75      0.78      6482
weighted avg       0.86      0.86      0.86      6482

--- k-Nearest Neighbours ---
Accuracy:   0.8959
Precision:  0.8569
Recall:     0.6291
F1:         0.7255
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4915                     149
Actual: Default (1)                           526                     892
Classification Report:
               precision    recall  f1-score   support

           0       0.90      0.97      0.94      5064
           1       0.86      0.63      0.73      1418

    accuracy                           0.90      6482
   macro avg       0.88      0.80      0.83      6482
weighted avg       0.89      0.90      0.89      6482

--- Decision Tree ---
Accuracy:   0.888
Precision:  0.7261
Recall:     0.7835
F1:         0.7537
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4645                     419
Actual: Default (1)                           307                    1111
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.92      0.93      5064
           1       0.73      0.78      0.75      1418

    accuracy                           0.89      6482
   macro avg       0.83      0.85      0.84      6482
weighted avg       0.89      0.89      0.89      6482


=== AFTER PCA (17 principal components) ===
--- Logistic Regression ---
Accuracy:   0.8607
Precision:  0.7609
Recall:     0.5296
F1:         0.6245
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4828                     236
Actual: Default (1)                           667                     751
Classification Report:
               precision    recall  f1-score   support

           0       0.88      0.95      0.91      5064
           1       0.76      0.53      0.62      1418

    accuracy                           0.86      6482
   macro avg       0.82      0.74      0.77      6482
weighted avg       0.85      0.86      0.85      6482

--- k-Nearest Neighbours ---
Accuracy:   0.8936
Precision:  0.85
Recall:     0.6234
F1:         0.7193
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4908                     156
Actual: Default (1)                           534                     884
Classification Report:
               precision    recall  f1-score   support

           0       0.90      0.97      0.93      5064
           1       0.85      0.62      0.72      1418

    accuracy                           0.89      6482
   macro avg       0.88      0.80      0.83      6482
weighted avg       0.89      0.89      0.89      6482

--- Decision Tree ---
Accuracy:   0.8511
Precision:  0.6559
Recall:     0.6721
F1:         0.6639
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       4564                     500
Actual: Default (1)                           465                     953
Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.90      0.90      5064
           1       0.66      0.67      0.66      1418

    accuracy                           0.85      6482
   macro avg       0.78      0.79      0.78      6482
weighted avg       0.85      0.85      0.85      6482


COMPARISON TABLE - BEFORE PCA:
                       Accuracy  Precision  Recall      F1
Logistic Regression     0.8645     0.7581  0.5592  0.6437
k-Nearest Neighbours    0.8959     0.8569  0.6291  0.7255
Decision Tree           0.8880     0.7261  0.7835  0.7537

COMPARISON TABLE - AFTER PCA:
                       Accuracy  Precision  Recall      F1
Logistic Regression     0.8607     0.7609  0.5296  0.6245
k-Nearest Neighbours    0.8936     0.8500  0.6234  0.7193
Decision Tree           0.8511     0.6559  0.6721  0.6639

F1-SCORE BEFORE VS AFTER PCA:
                       F1 Before PCA  F1 After PCA  Change
Logistic Regression          0.6437        0.6245 -0.0192
k-Nearest Neighbours         0.7255        0.7193 -0.0062
Decision Tree                0.7537        0.6639 -0.0898

======================================================================
PART F: MODEL OPTIMISATION
======================================================================

Best model from Part E: Decision Tree (Before PCA) -> F1 = 0.7537

Best parameters: {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}
Best CV F1-score: 0.7963

=== TUNED MODEL RESULTS ===
--- Decision Tree (Tuned) ---
Accuracy:   0.9287
Precision:  0.9492
Recall:     0.7123
F1:         0.8139
Confusion Matrix (top-left = TN, top-right = FP, bottom-left = FN, bottom-right = TP):
                        Predicted: No Default (0)  Predicted: Default (1)
Actual: No Default (0)                       5010                      54
Actual: Default (1)                           408                    1010
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.99      0.96      5064
           1       0.95      0.71      0.81      1418

    accuracy                           0.93      6482
   macro avg       0.94      0.85      0.88      6482
weighted avg       0.93      0.93      0.92      6482


DECISION TREE: BEFORE VS AFTER TUNING (test set):
            Before Tuning  After Tuning  Change
Accuracy          0.8880        0.9287  0.0407
Precision         0.7261        0.9492  0.2231
Recall            0.7835        0.7123 -0.0712
F1                0.7537        0.8139  0.0601

TRAIN VS TEST F1 (overfitting check):
                        Train F1  Test F1  Gap (Train - Test)
Untuned Decision Tree    1.0000   0.7537              0.2463
Tuned Decision Tree      0.8207   0.8139              0.0068
Tree depth - untuned: 33 | tuned: 10