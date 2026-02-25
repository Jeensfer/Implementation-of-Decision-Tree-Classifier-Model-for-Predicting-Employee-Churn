# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and preprocess it by encoding categorical features.
2. Separate the dataset into input features (X) and target variable (left).
3. Split the data into training and testing sets and train the Decision Tree classifier.
4. Predict employee churn using the test data and evaluate the model using accuracy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Jeensfer Jo
RegisterNumber:  212225240058
*/
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
df = pd.read_csv("Employee.csv") 
df.columns = df.columns.str.strip()

# Step 3: Check Data
print(df.head())
print(df.info())

# Step 4: Encode Categorical Variables
le_dept = LabelEncoder()
le_salary = LabelEncoder()

df['Departments'] = le_dept.fit_transform(df['Departments'])
df['salary'] = le_salary.fit_transform(df['salary'])

# Step 5: Define Features and Target
X = df.drop('left', axis=1)   # Features
y = df['left']               

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 7: Create Decision Tree Model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

# Step 8: Train Model
model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
<img width="1152" height="685" alt="image" src="https://github.com/user-attachments/assets/4d28aec9-7adf-4685-ade1-ba24c0de4e7a" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
