#Data Loading
import numpy as np

#Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Modelling
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

#Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score


import pandas as pd
import os

os.chdir("C:/Users/Syafiq Irfan/OneDrive/Desktop/Projects/Monkeypox")

Worldwide_cases = pd.read_csv('Monkey_Pox_Cases_Worldwide.csv')
Worldwide_cases.head()

Worldwide_cases.isna().sum()

-------------------------------------------------------------------------------

#Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

Confirmed_Cases = Worldwide_cases.loc[:,["Country", "Confirmed_Cases"]].sort_values(by = 'Confirmed_Cases', ascending = False).head(10)

c = ['r', 'y', 'g', 'b', 'c', 'k','olive', 'gray', 'pink', 'maroon']
bar = Confirmed_Cases.plot.bar(x="Country", y="Confirmed_Cases", alpha=1, color=c)
bar.set_xlabel("Country", fontweight ='bold')
bar.set_ylabel("Confirmed_Cases", fontweight ='bold')
plt.title('Number of Confirmed Cases', fontweight='bold')
plt.gca ().get_legend ().remove ()


Hospitalized_Cases = Worldwide_cases.loc[:,["Country", "Hospitalized"]].sort_values(by = 'Hospitalized', ascending = False).head(10)

c = ['r', 'y', 'g', 'b', 'c', 'k','olive', 'gray', 'pink', 'maroon']
bar = Hospitalized_Cases.plot.bar(x="Country", y="Hospitalized", alpha=1, color=c)
bar.set_xlabel("Country", fontweight ='bold')
bar.set_ylabel("Hospitalized Cases", fontweight ='bold')
plt.title('Number of Hospitalized Cases', fontweight='bold')
plt.gca ().get_legend ().remove ()


import plotly.express as px

def top10plots(col = None):
    #Sorting the Dataset
    Worldwide_cases_sorted = Worldwide_cases.sort_values(by=col,ascending=False).reset_index()
    #Getting the Top10
    top10 = Worldwide_cases_sorted[:10]
    # Plotting the Top10
    label_text = ' '.join(col.split('_'))
    labeldict = {'size':'15','weight':'3'}
    titledict = {'size':'20','weight':'3'}
    fig = px.bar(x='Country',
                 y=col,
                 data_frame=top10,
                 labels=['Country',label_text],
                 color=col,
                 color_continuous_scale='electric',
                 text_auto=True,
                 title=f'Top 10 Countries based on {label_text}')
    fig.show()

top10plots(col='Travel_History_Yes')
top10plots(col='Travel_History_No')


-------------------------------------------------------------------------------

# Correlations

b = ['blue']
Cases_Hospitalizations_Corr = Worldwide_cases.plot.scatter('Confirmed_Cases', 'Hospitalized', figsize = (10,6), color = b)
Cases_Hospitalizations_Corr.set_xlabel("Confirmed Cases", fontweight ='bold')
Cases_Hospitalizations_Corr.set_ylabel("Hopitalizations", fontweight ='bold')
plt.title('Confirmed Cases vs Hopitalized', fontweight='bold')

r = ['red']
Cases_Travellersyes_Corr = Worldwide_cases.plot.scatter('Confirmed_Cases', 'Travel_History_Yes', figsize = (10,6), color = r)
Cases_Travellersyes_Corr.set_xlabel("Confirmed Cases", fontweight ='bold')
Cases_Travellersyes_Corr.set_ylabel("Travel History", fontweight ='bold')
plt.title('Confirmed Cases vs Travel History', fontweight='bold')

g = ['green']
Cases_Travellersno_Corr = Worldwide_cases.plot.scatter('Confirmed_Cases', 'Travel_History_No', figsize = (10,6), color = g)
Cases_Travellersno_Corr.set_xlabel("Confirmed Cases", fontweight ='bold')
Cases_Travellersno_Corr.set_ylabel("No Travel History", fontweight ='bold')
plt.title('Confirmed Cases vs No Travel History', fontweight='bold')

# Heatmap

Variables = Worldwide_cases.loc[:,['Confirmed_Cases','Hospitalized','Travel_History_Yes','Travel_History_No']]
sns.set(font_scale=0.8)
ax = sns.heatmap(Variables.corr(), linewidth = 1, cmap="RdYlBu", square=True)
ax.figure.tight_layout()
plt.title('Correlation Heatmap')
plt.show()









