import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import arabic_reshaper
from bidi.algorithm import get_display

# Load data
data = pd.read_csv('Datasets/customer_stories.csv')  # Replace with your data file

# Preprocess Arabic text
nltk.download('punkt')

def preprocess_arabic_text(text):
    if pd.isna(text):
        text = ''  # Handle NaN values by replacing them with an empty string
    reshaped_text = arabic_reshaper.reshape(str(text))  # Ensure the text is a string
    bidi_text = get_display(reshaped_text)
    tokens = word_tokenize(bidi_text)
    return ' '.join(tokens)

data['story'] = data['story'].apply(preprocess_arabic_text)

# Replace missing action codes with 'NULL'
data[['action_code1', 'action_code2', 'action_code3']] = data[['action_code1', 'action_code2', 'action_code3']].fillna('NULL')

# Load action codes (English descriptions)
action_codes = pd.read_csv('Datasets/action_codes.csv')  # Replace with your action codes file

# Convert action codes to numerical values, including 'NULL'
action_code_map = {code: idx for idx, code in enumerate(action_codes['action_code'].unique().tolist() + ['NULL'])}
inverse_action_code_map = {v: k for k, v in action_code_map.items()}

data['action_code1'] = data['action_code1'].map(action_code_map)
data['action_code2'] = data['action_code2'].map(action_code_map)
data['action_code3'] = data['action_code3'].map(action_code_map)

# Extract features and labels
X = data['story']  # Stories or narratives about bank account events
y = data[['action_code1', 'action_code2', 'action_code3']]  # Multiple action codes

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train model using MultiOutputClassifier
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Function to predict actions for a new story
def predict_actions(story):
    story_preprocessed = preprocess_arabic_text(story)
    story_vect = vectorizer.transform([story_preprocessed])
    predicted_codes = model.predict(story_vect)[0]
    return [inverse_action_code_map[code] for code in predicted_codes]

# Load the new stories CSV file
new_stories = pd.read_csv('Datasets/Sheet2.csv')  

# Initialize lists to store the predicted codes
predicted_action_code1 = []
predicted_action_code2 = []
predicted_action_code3 = []

# Predict action codes for each story
for story in new_stories['story']:
    predicted_codes = predict_actions(story)
    predicted_action_code1.append(predicted_codes[0])
    predicted_action_code2.append(predicted_codes[1])
    predicted_action_code3.append(predicted_codes[2])

# Add the predicted codes to the new_stories DataFrame
new_stories['predicted_action_code1'] = predicted_action_code1
new_stories['predicted_action_code2'] = predicted_action_code2
new_stories['predicted_action_code3'] = predicted_action_code3

# Save the results to a new CSV file
new_stories.to_csv('Datasets/Sheet2(Predictions).csv', index=False)  

print("Predictions saved to Sheet2(Predictions).csv")
