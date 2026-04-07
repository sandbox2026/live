import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

data = pd.read_csv('/content/ox.csv')
print("Data is loaded :",data.head())

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

LE = LabelEncoder()
x['Outlook'] = LE.fit_transform(x['Outlook'])
x['Temperature'] = LE.fit_transform(x['Temperature'])
x['Humidity'] = LE.fit_transform(x['Humidity'])
x['Windy'] = LE.fit_transform(x['Windy'])

y = LE.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

cl = GaussianNB()
cl.fit(x_train,y_train)

y_pred = cl.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["No","Yes"])

disp.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


plt.scatter(range(len(y_test)),y_test,color='blue',label="Actual_Values",marker="o")
plt.scatter(range(len(y_pred)),y_pred,color='red',label="Predicted_Values",marker="x")
plt.xlabel("Test Sample Index")
plt.ylabel("Play Tennis (0-NO 1-Yes)")
plt.title("Actual vs Predicted Values")

plt.legend()
plt.grid(True)
plt.show()
