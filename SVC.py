import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# read the preprocessed file
df = pd.read_csv("sentimentdataset_stopwords_lemmatized.csv")

X = df['Message']
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transform the training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# to convert it to a data frame
# df_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

svc = LinearSVC(max_iter=100000, dual=False)

# Define the parameter grid for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'loss': ['squared_hinge'],
    'tol': [1e-3, 1e-2, 1e-1, 1],
    'class_weight': [None, 'balanced']
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(svc, param_grid, scoring='accuracy', error_score='raise', cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters found by grid search
print("Best Parameters: ", grid_search.best_params_)

# Predict on the testing set with the best model
y_pred = grid_search.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Testing Set: {:.2f}%".format(accuracy * 100))

# Display additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))