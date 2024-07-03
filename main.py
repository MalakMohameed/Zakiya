import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import arabic_reshaper
from bidi.algorithm import get_display

# Load data
data = pd.read_csv('Datasets\\customer_stories.csv')  # Replace with your data file

# Preprocess Arabic text
nltk.download('punkt')

def preprocess_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    tokens = word_tokenize(bidi_text)
    return ' '.join(tokens)

data['story'] = data['story'].apply(preprocess_arabic_text)

# Extract features and labels
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

# Function to predict action for a new story
def predict_action(story):
    story_preprocessed = preprocess_arabic_text(story)
    story_vect = vectorizer.transform([story_preprocessed])
    predicted_code = model.predict(story_vect)[0]
    action_description = action_codes.loc[action_codes['action_code'] == predicted_code, 'action_description'].values[0]
    return predicted_code, action_description

# Load action codes (English descriptions)
action_codes = pd.read_csv('Datasets\\action_codes.csv')  # Replace with your action codes file

# Test the model with multiple stories
def print_prediction(story):
    predicted_code, action_description = predict_action(story)
    print(f"Story: {story}")
    print(f"Predicted Action Code: {predicted_code}, Action Description: {action_description}")
    print()

# Define the stories (in Arabic)
story1 = "تم رفض بطاقة الائتمان الخاصة بالعميل أثناء عملية شراء دولية."
story2 = "أبلغ العميل عن معاملة غير مصرح بها على حسابه."
story3 = "انخفض رصيد حساب العميل إلى أقل من الحد الأدنى المطلوب."
story4 = "قام العميل بعملية شراء كبيرة في متجر فاخر."
story5 = "كان هناك محاولة تسجيل دخول مشبوهة من عنوان IP أجنبي."
story6 = "استخدمت بطاقة الخصم الخاصة بالعميل في دولة مختلفة."
story7 = "تلقى العميل إشعارًا بشأن دفع بطاقة الائتمان المتأخرة."
story8 = "تم رفض المعاملة بسبب نقص الأموال."
story9 = "تم وضع علامة على حساب العميل للنشاط الاحتيالي المحتمل."
story10 = "فات العميل سداد فاتورة بطاقة الائتمان لدورة الفوترة الأخيرة."

# Predict and print actions for each story
print_prediction(story1)
print_prediction(story2)
print_prediction(story3)
print_prediction(story4)
print_prediction(story5)
print_prediction(story6)
print_prediction(story7)
print_prediction(story8)
print_prediction(story9)
print_prediction(story10)
