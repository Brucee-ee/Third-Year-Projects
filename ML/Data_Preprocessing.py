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

#**TASK 3.1: TRAIN THE MODELS**

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Train k-NN (k = 5)
print("Training k-NN")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("k-NN trained, training accuracy:", round(knn.score(X_train, y_train), 3))
 
# Step 2: Train SVM
print("Training SVM")
svm = SVC()
svm.fit(X_train, y_train)
print("SVM trained, training accuracy:", round(svm.score(X_train, y_train), 3))
 
# Step 3: Train LDA
print("Training LDA")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print("LDA trained, training accuracy:", round(lda.score(X_train, y_train), 3))

print("All three models trained")

#**Task 3.2: TUNE THE HYPERPARAMETERS**

from sklearn.model_selection import cross_val_score

# Step 1: Tune k-NN
print("Tuning k-NN")
best_k = 0
best_score = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    score = scores.mean()
    print("k =", k, "Average Accuracy:", round(scores.mean(), 3))

    if score.mean() > best_k_score:
        best_k_score = score.mean()
        best_k = k

print("Best k:", best_k, "with Average Accuracy:", round(best_k_score, 3))

# Step 2: Tune SVM
print("Tuning SVM")
best_kernal = ""
best_Kernal_score = 0

for kernal in ["linear", "poly", "rbf", "sigmoid"]:
    svm = SVC(kernel=kernal)
    scores = cross_val_score(svm, X_train, y_train, cv=5)
    score = scores.mean()
    print("Kernal =", kernal, "Average Accuracy:", round(scores.mean(), 3))

    if score.mean() > best_Kernal_score:
        best_Kernal_score = score.mean()
        best_kernal = kernal

print("Best kernal:", best_kernal, "with Average Accuracy:", round(best_Kernal_score, 3))

# Step 3: LDA has no hyperparameters to tune
print("\nLDA: no hyperparameters to tune")

# Step 4: Retrain k-NN and SVM on the full training set using the best settings
