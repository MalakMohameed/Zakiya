import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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

# Replace missing action codes with 'NULL'
data[['action_code1', 'action_code2', 'action_code3']] = data[['action_code1', 'action_code2', 'action_code3']].fillna('NULL')

# Load action codes (English descriptions)
action_codes = pd.read_csv('Datasets\\action_codes.csv')  # Replace with your action codes file

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

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
# for i, col in enumerate(y.columns):
#     print(f"Classification Report for {col}:")
#     print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Function to predict actions for a new story
def predict_actions(story):
    story_preprocessed = preprocess_arabic_text(story)
    story_vect = vectorizer.transform([story_preprocessed])
    predicted_codes = model.predict(story_vect)[0]
    action_descriptions = [action_codes.loc[action_codes['action_code'] == inverse_action_code_map[code], 'action_description'].values[0] if inverse_action_code_map[code] != 'NULL' else 'No Action' for code in predicted_codes]
    return [inverse_action_code_map[code] for code in predicted_codes], action_descriptions

# Test the model with multiple stories
def print_prediction(story):
    predicted_codes, action_descriptions = predict_actions(story)
    print(f"Story: {story}")
    for code, desc in zip(predicted_codes, action_descriptions):
        print(f"Predicted Action Code: {code}, Action Description: {desc}")
    print()

# Define the stories (in Arabic)
story1 = "مطلوب تحديث بيانات العميل:مطلوب تحد:يث بيانات العميل:Major customer now"
story2 = "عدم سحب تمويل عقارى"
story3 = "وفاة العميل"
story4 = "عدم السحب"
story5 = "Major customer now:مطلوب تحديث بيان:ات العميل"
story6 = "وثيقة تأمين لأول مرةبمبلغ 90000 عدم سحب تمويل عقاري"
story7 = ".عدم السحب من الحساب قبل تحديث البي:انات و استيقاء المستندات المطلوبة ط:بقا لمتطلبات الفاتكاالرجوع الى قسم خدمه العملاء لتغيير :عنوان المراسلاتتحديث فاتيكا عدم ال"
story8 = ".قرض 10000جم يسدد على 12 قسط بقسط ش:هرى 970جم تقريبا بكفالة  عصمت مصطف:ى محمد متولى من 2362018 حتى 235:2019صرف 2352018"
story9 = "قرض حكومى قدره41721جم لمدة 60 شهر م:ن 31012016 حتى 31122020 قسط شهر:ى 960جم"
story10 = "يؤول رصيد حساب التوفير بالجنيه المص:ري بعد وفاه العمليه لمصلحه الابن عم:ر عبدالسميع محمد يوسف والابن احمد ع:بدالسميع محمد يوسف بالتساوي بينهما : وكذلك مايستجد من ارص"
story11 = "؛ و قروض الحكومة عدم السحب من الحس:اب لعدم سداد الاقساط.العميل متوفى بتاريخ 17082020 .وتم: ابلاغنا بتاريخ 06092020"
story12 = "وفاة العميل 12091985 .:مطلوب تحديث بيانات العميل:مطلوب تح:ديث بيانات العميل"

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
print_prediction(story11)
print_prediction(story12)
