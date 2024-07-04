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
story1 = "مطلوب تحديث بيانات العميل:مطلوب تحد:يث بيانات العميل:Major customer now"
story2 = "عدم سحب تمويل عقارى"
story3 = "وفاة العميل"
story4 = "عدم السحب"
story5 = "Major customer now:مطلوب تحديث بيان:ات العميل"
story6 = "‘عدم السحب الابعدسداد المتأخرات"
story7 = ".عدم السحب من الحساب قبل تحديث البي:انات و استيقاء المستندات المطلوبة ط:بقا لمتطلبات الفاتكاالرجوع الى قسم خدمه العملاء لتغيير :عنوان المراسلاتتحديث فاتيكا عدم ال"
story8 = ".قرض 10000جم يسدد على 12 قسط بقسط ش:هرى 970جم تقريبا بكفالة  عصمت مصطف:ى محمد متولى من 2362018 حتى 235:2019صرف 2352018"
story9 = "قرض حكومى قدره41721جم لمدة 60 شهر م:ن 31012016 حتى 31122020 قسط شهر:ى 960جم"
story10 = "يؤول رصيد حساب التوفير بالجنيه المص:ري بعد وفاه العمليه لمصلحه الابن عم:ر عبدالسميع محمد يوسف والابن احمد ع:بدالسميع محمد يوسف بالتساوي بينهما : وكذلك مايستجد من ارص"
story11 = "؛ و قروض الحكومة عدم السحب من الحس:اب لعدم سداد الاقساط.العميل متوفى بتاريخ 17082020 .وتم: ابلاغنا بتاريخ 06092020"
story12 = "وثيقة تأمين لأول مرةبمبلغ 90000 عدم سحب تمويل عقاري"
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
