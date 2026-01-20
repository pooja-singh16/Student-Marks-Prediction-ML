# Student-Marks-Prediction-ML
My first Machine Learning project using Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [10,20,30,40,50,60,70,80,85,95]
}

df = pd.DataFrame(data)
df


X = df[["Hours"]]   # Input (study hours)
y = df["Marks"]    # Output (marks)


model = LinearRegression()
model.fit(X, y)
print("Model trained successfully!")

prediction = model.predict([[6]])
print("If student studies 6 hours, predicted marks =", prediction[0])

plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
