# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#**TASK 1: Data Preprocessing**

# Clean it up in command prompt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Step 1: Load the dataset
# Step 1.1: Display the first few rows of the dataset

data = pd.read_csv("winequality-white.csv", sep=";")
print(data.head())

# Step 2: Find/Check for missing values
print(data.isnull().sum())

# Step 2.1: No missing vlalues found, drop just in case
data = data.dropna()

# Step 3: Separate features and target variable
X = data.drop("quality", axis=1)
y = data["quality"]

# Step 4: Normalise the features, all between 0 and 1
scaler = MinMaxScaler()
x = scaler.fit_transform(X)
x = pd.DataFrame(x, columns=data.columns[:-1])
print(x.head())

# Step 5: Make quality 1 = good, 0 = bad
y =[]
for q in data["quality"]:
    if q >= 6:
        y.append(1)
    else:
        y.append(0)

y = pd.Series(y)
print(y.value_counts())

#**TASK 2: TRAIN TEST SPLIT**

# Step 1: Split the dataset into training 70% and testing sets 30%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Step 2: Diplay the rows

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
