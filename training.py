from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import export_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
from agent import SofiaAgent
import numpy as np

version = 'testing'

# Read the survey data into a pandas dataframe
df = pd.read_csv('datasets/dataset.csv')

# Load the agent to fit it
agent = SofiaAgent(model_path=None, vectorizer_path=None)

# Clean and preprocess the questions using the agent's method
df['Cleaned_Question'] = df['Question'].apply(agent.clean_question_text)

# Convert the questions to lowercase and tokenize them
vectorizer = TfidfVectorizer(lowercase=True, tokenizer=agent.extract_features_tokenizer)

# Transform the questions into feature vectors
# (e.g.) X[0] => asesor dispon horari vespertin - but in numeric way [(0, 20), (0, 82), (0, 124), (0, 251)]
X = vectorizer.fit_transform(df['Cleaned_Question'])
print(vectorizer.vocabulary_)
# Assign labels to each question
y = df['Label']

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"{X_train.shape[0]} training samples")

# Define the hyperparameter grid to search over
# tree_param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [3, 5, 7, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_leaf_nodes': [None, 5, 10, 20],
#     'class_weight': [None, 'balanced'],
#     'splitter': ['best', 'random']
# }

# mb_param_grid = {
#     'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
# }

svm_param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 20, 100],
    'class_weight': [None, 'balanced']
}

# Create the GridSearchCV object with the classifier
grid_search = GridSearchCV(SVC(probability=True), svm_param_grid, cv=10, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Train the final model on all the available data with the best hyperparameters
clf = SVC(**grid_search.best_params_, probability=True)
clf.fit(X_train, y_train)

# Perform k-fold cross-validation to estimate the model's performance
cv_scores = cross_val_score(clf, X, y, cv=10)  # X and y should be your full dataset

# Print the mean and standard deviation of the cross-validation scores
print("Cross-validation mean accuracy:", cv_scores.mean())
print("Cross-validation standard deviation:", cv_scores.std())

# Decision Tree rules
# tree_rules = export_text(clf, feature_names=vectorizer.get_feature_names_out().tolist())

# Write the decision tree rules to a text file
# with open('decision_tree_rules.txt', 'w') as f:
#    f.write(tree_rules)
    
# Evaluate the performance of the final model on the testing set
y_pred_train = clf.predict(X_train)
report_train = classification_report(y_train, y_pred_train, zero_division=0)

# Print the report and accuracy for the model
print("Training set report:")
print(report_train)
accuracy_train = clf.score(X_train, y_train)
print(f"Training set accuracy: {accuracy_train}")

# Evaluate the performance of the final model on the testing set
y_pred_test = clf.predict(X_test)
report_test = classification_report(y_test, y_pred_test, zero_division=0)

# Print the report and accuracy for the final model
print("Final model report:")
print(report_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Calculate predicted probabilities
predicted_probabilities = clf.predict_proba(X_test)
# Find indices of samples with max probability less than 0.5
low_confidence_indices = np.where(np.max(predicted_probabilities, axis=1) < 0.5)
# Retrieve the original text samples from the test set
low_confidence_samples = X_test[low_confidence_indices]
# Convert the samples back to text using the inverse_transform method of the vectorizer
low_confidence_text = vectorizer.inverse_transform(low_confidence_samples)
# Retrieve the max probabilities for the low-confidence samples
low_confidence_probabilities = np.max(predicted_probabilities[low_confidence_indices], axis=1)
# Retrieve the predicted labels for the low-confidence samples
low_confidence_labels = y_pred_test[low_confidence_indices]
# Print the low confidence text samples along with their confidence levels and predicted labels
for idx, (sample, confidence, label) in enumerate(zip(low_confidence_text, low_confidence_probabilities, low_confidence_labels)):
    print(f"Sample {idx}: {' '.join(sample)}")
    print(f"Confidence: {confidence}")
    print(f"Predicted Label: {label}\n")

# Export the vectorizer and the trained model
joblib.dump(vectorizer, 'models/vectorizer.joblib')
joblib.dump(clf, f'models/model-{version}.joblib')