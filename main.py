import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# datasets
stories_file = "Datasets\\customer_stories.csv"
actions_file = "Datasets\\action_codes.csv"

stories = pd.read_csv(stories_file)
actions = pd.read_csv(actions_file)

#just to see if the data is being loaded  correctly 
#######################################3
# print("Customer Stories Data:")
# print(stories.head())
# print("\nAction Codes Data:")
# print(actions.head())

data = pd.merge(stories, actions, on='id')

# print("\nMerged Data:")
# print(data.head())
#####################################



# Check if 'action_code' column exists
if 'action_code' not in data.columns:
    raise KeyError("The 'action_code' column is missing from the merged data.")

# Preprocess data
X = data['story']  # Stories or narratives about bank account events
y = data['action_code']  # Action codes indicating what action was taken

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
#print(classification_report(y_test, y_pred))

# Prediction fucntion
def predict_action(new_story):
    new_story_vect = vectorizer.transform([new_story])
    predicted_code = model.predict(new_story_vect)[0]
    action_description = actions[actions['action_code'] == predicted_code]['action_description'].values[0]
    return predicted_code, action_description

# Example usage

story1 = "Customer's credit card was declined during an international purchase."
story2 = "Customer reported unauthorized transaction on their account."
story3 = "Customer's account balance fell below the minimum required threshold."
story4 = "Customer made a large purchase at a luxury store."
story5 = "There was a suspicious login attempt from a foreign IP address."
story6 = "Customer's debit card was used in a different state."
story7 = "Customer received a notification about an overdue credit card payment."
story8 = "Transaction declined due to insufficient funds."
story9 = "Customer's account was flagged for potential fraudulent activity."
story10 = "Customer's credit card payment was missed for the last billing cycle."

def print_pre(story):
    predicted_code, action_description = predict_action(story)
    print(f"Story: {story}")
    print(f"Predicted Action Code: {predicted_code}, Action Description: {action_description}")
    print()

# Predict and print actions for each story
print_pre(story1)
print_pre(story2)
print_pre(story3)
print_pre(story4)
print_pre(story5)
print_pre(story6)
print_pre(story7)
print_pre(story8)
print_pre(story9)
print_pre(story10)
