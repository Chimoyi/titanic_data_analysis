#importing the necessary libraries
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import numpy as py
import seaborn as sns

st.title("Titanic Dataset Analysis")
st.write("This app analyzes the Titanic dataset and displays various visualizations.")
#load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    return df

data = load_data()

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

#shows the survival rate by gender
gender_survival = data.groupby('Sex')['Survived'].mean().reset_index()
fig_gender_survival = px.bar(gender_survival, x='Sex', y='Survived', title='Survival Rate by Gender')
st.plotly_chart(fig_gender_survival)

#shows the survival rate by passenger class
pclass_survival = data.groupby('Pclass')['Survived'].mean().reset_index()
fig_pclass_survival = px.bar(pclass_survival, x='Pclass', y='Survived', title='Survival Rate by Passenger Class')
st.plotly_chart(fig_pclass_survival)

#shows the age distribution of the passengers
fig_age_distribution = px.histogram(data, x='Age', nbins=50, title='Age Distribution of Passengers')
st.plotly_chart(fig_age_distribution)

#shows the age distribution by survival status
fig_age_survival = px.histogram(data, x='Age', color='Survived', nbins=50, title='Age Distribution by Survival Status')
st.plotly_chart(fig_age_survival)