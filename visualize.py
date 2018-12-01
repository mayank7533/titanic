# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt 


df=pd.read_csv("train.csv")

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived,df.Age,alpha=0.1)
plt.title("Age wrt survival")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.1)
plt.title("Pclass") 

plt.subplot2grid((2,3),(1,0),colspan=4)
for x in [1,2,3]:
    df.Survived[df.Pclass==x].plot(kind="kde")
plt.title("class wrt survived")
plt.legend(("1st","2nd","3rd"))



