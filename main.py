import pandas as pd
import tensorflow  as tf
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("Datasets\customer_stories.csv")  # Replace with your data file path

# Preprocess data
x = data.drop(columns=['action_code'])  # Stories or narratives about bank account events
y = data['action_code']  # Action codes indicating what action was taken

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Predictions
model = tf.keras.models.Sequetial()

model.add(tf.keras.layers.Dense(25,input_shape=X_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(25,activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))


model.compile(optimizer ='admin', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, epochs=1000)