import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset
df = pd.read_csv("loan_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


# 2. Remove unnecessary column
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)


# 3. Understand the data
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary (numeric columns):")
print(df.describe())

print("\nStatistical Summary (categorical columns):")
print(df.describe(include="object"))

print("\nColumn Data Types:")
print(df.dtypes)


# 4. Check missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())


# 5. Check duplicate records
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Drop duplicates if any exist
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("Duplicates removed. New shape:", df.shape)


# 6. Handle missing values
for column in df.columns:

    # Numeric columns
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())

    # Categorical / string columns
    else:
        df[column] = df[column].fillna(df[column].mode()[0])


# 7. Simple EDA
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.show()


# 8. Convert categorical columns into numbers
label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = label_encoder.fit_transform(df[column].astype(str))


# 9. Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# 10. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 11. Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# 12. Train model
model.fit(X_train, y_train)


# 13. Make predictions
y_pred = model.predict(X_test)


# 14. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# 15. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 16. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# 17. Predict a new loan application
new_applicant = pd.DataFrame({
    "Gender": [1],
    "Married": [1],
    "Dependents": [0],
    "Education": [0],
    "Self_Employed": [0],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [1000],
    "LoanAmount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1],
    "Property_Area": [2]
})

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("\nLoan Application Status: APPROVED")
else:
    print("\nLoan Application Status: REJECTED")


# 18. Save the trained model as a pickle file
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as loan_model.pkl")