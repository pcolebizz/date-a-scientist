import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Create your df here:
df = pd.read_csv("profiles.csv")

#Explore the Income Data
#print(df.income.value_counts())

# Visualize Income Data
plt.hist(df.income, bins=40)
plt.xlabel("Income")
plt.ylabel("Number of People")
plt.xlim(20000, 100000)
plt.ylim(50, 3000)
plt.show()

#Explore the Education Data
#print(df.education.value_counts())

education_mapping = {"space camp": 0, "high school": 1, "two-year college": 2, "working on two-year college": 3, "working on college/university": 4, "graduated from college/university": 5, "graduated from masters program": 6}
df["education_code"] = df.education.map(education_mapping)

#remove nans
df["education_code"] = df["education_code"].replace(np.nan, 0, regex=True).astype(int)
print(df.education_code.value_counts())

#remove NANs from income and education_code
edu_v_income = df.loc[:,['education_code', 'income']]
edu_v_income.dropna(inplace=True)

edu_v_income = edu_v_income[(edu_v_income[['income']] != -1).all(axis=1)]
train, test = train_test_split(edu_v_income, train_size=0.8, test_size = 0.2)

Xtr = train[['education_code']]
ytr = train[['income']]
regr = LinearRegression()
regr.fit(Xtr,ytr)
y_predicted = regr.predict(Xtr)
X = test[['education_code']]
y = test[['income']]
print('Education v. Income Regression Score: ', regr.score(X,y))

plt.xlabel('education')
plt.ylabel('Income')
plt.scatter(X,y,alpha=0.2)
plt.plot(Xtr, y_predicted)
plt.xlim(0,6)
plt.ylim(20000,110000)
#plt.show()

