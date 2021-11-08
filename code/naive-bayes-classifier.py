import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

play_dataset = pd.read_csv("../dataset/playbasedweather.csv")
play_dataset.head()

num_encode = LabelEncoder()
play_dataset['Outlook'] = num_encode.fit_transform(play_dataset['Outlook'])
play_dataset['Temperature'] = num_encode.fit_transform(play_dataset['Temperature'])
play_dataset['Humidity'] = num_encode.fit_transform(play_dataset['Humidity'])
play_dataset['Windy'] = num_encode.fit_transform(play_dataset['Windy'])
play_dataset['Play'] = num_encode.fit_transform(play_dataset['Play'])

label_features = ["Outlook","Temperature","Humidity","Windy"]
label_target = "Play"

features_train, features_test, target_train, target_test = train_test_split(play_dataset[label_features], play_dataset[label_target], 
                                                                test_size=0.33, random_state=54)


model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)

# Printing accuracy
print(accuracy)

# Label Encoding for Rainy, Cool, High, False
# Rainy = 1
# Cool = 0
# High = 0
# False = 0

print('Hasil' + str(model.predict([[1,0,0,0]])))