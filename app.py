### Who's a LinkedIn User? Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


st.markdown("# Who's a LinkedIn User?")
st.markdown("This app uses a machine learning model (logistic regression) to predict whether a person uses LinkedIn, based on their income, education, parent status, marital status, gender, and age. Change the inputs below and the prediction updates in real time.")


###### 1 Read in and clean the data, and train the model (same steps as Part 1)
# Cached so this only runs once, not every time an input changes

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


@st.cache_resource
def train_model():
    s = pd.read_csv("social_media_usage.csv")

    ss = pd.DataFrame({
        "sm_li": clean_sm(s["web1h"]),
        "income": np.where(s["income"] > 9, np.nan, s["income"]),
        "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent": np.where(s["par"] > 2, np.nan,
                           np.where(s["par"] == 1, 1, 0)),
        "married": np.where(s["marital"] > 6, np.nan,
                            np.where(s["marital"] == 1, 1, 0)),
        "female": np.where(s["gender"] > 3, np.nan,
                           np.where(s["gender"] == 2, 1, 0)),
        "age": np.where(s["age"] > 98, np.nan, s["age"])
    })

    ss = ss.dropna()

    y = ss["sm_li"]
    X = ss[["income", "education", "parent", "married", "female", "age"]]

    X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                        y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=987)

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)

    return ss, lr, X_test, y_test


ss, lr, X_test, y_test = train_model()

###### 3 User inputs

st.markdown("## Enter a person's information")

income_labels = ["Less than $10,000",
                 "$10,000 to under $20,000",
                 "$20,000 to under $30,000",
                 "$30,000 to under $40,000",
                 "$40,000 to under $50,000",
                 "$50,000 to under $75,000",
                 "$75,000 to under $100,000",
                 "$100,000 to under $150,000",
                 "$150,000 or more"]

education_labels = ["Less than high school",
                    "High school incomplete",
                    "High school graduate",
                    "Some college, no degree",
                    "Two-year associate degree",
                    "Four-year college degree (Bachelor's)",
                    "Some postgraduate schooling, no degree",
                    "Postgraduate or professional degree"]

ages = [each for each in range(18, 98)]

income_choice = st.selectbox(label="Household income", options=income_labels)
education_choice = st.selectbox(label="Highest level of education", options=education_labels)
parent_choice = st.selectbox(label="Parent of a child under 18 living at home?", options=("No", "Yes"))
married_choice = st.selectbox(label="Married?", options=("No", "Yes"))
gender_choice = st.selectbox(label="Gender", options=("Female", "Male", "Other"))
age = st.selectbox(label="Age", options=ages)


# Convert the answers into the numbers the model uses
def yes_no(choice):
    if choice == "Yes":
        return 1
    else:
        return 0


income = income_labels.index(income_choice) + 1        # 1 to 9
education = education_labels.index(education_choice) + 1  # 1 to 8
parent = yes_no(parent_choice)
married = yes_no(married_choice)

if gender_choice == "Female":
    female = 1
else:
    female = 0


###### 4 Make the prediction

# Features in the same order as X: income, education, parent, married, female, age
person = [income, education, parent, married, female, age]

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])
prob_user = probs[0][1]

st.markdown("## Prediction")

if predicted_class[0] == 1:
    st.markdown("### This person is predicted to be a **LinkedIn user**")
else:
    st.markdown("### This person is predicted to **not** be a LinkedIn user")

st.markdown(f"### Probability that this person uses LinkedIn: {round(prob_user * 100, 1)}%")

prob_df = pd.DataFrame({
    "Outcome": ["Uses LinkedIn", "Does not use LinkedIn"],
    "Probability": [prob_user, 1 - prob_user]
})

prob_plot = alt.Chart(prob_df).mark_bar().encode(
    x=alt.X("Probability", scale=alt.Scale(domain=[0, 1])),
    y="Outcome",
    color="Outcome",
    tooltip=["Outcome", "Probability"]).properties(title="Predicted probability for this person")

st.altair_chart(prob_plot, use_container_width=True)


###### 5 How the probability changes with age for this person

st.markdown("## How would the prediction change with age?")
st.markdown("Same person (same income, education, parent, marital, and gender inputs) at every age from 18 to 97. The selected age is highlighted.")

profiles = [[income, education, parent, married, female, each] for each in ages]
age_probs = lr.predict_proba(profiles)

prob_by_age = pd.DataFrame({
    "Age": ages,
    "Probability of using LinkedIn": [each[1] for each in age_probs]
})

prob_by_age["Point"] = np.where(prob_by_age["Age"] == age, "Selected age", "Other ages")

age_plot = alt.Chart(prob_by_age).mark_circle(size=60).encode(
    x="Age",
    y=alt.Y("Probability of using LinkedIn", scale=alt.Scale(domain=[0, 1])),
    color="Point",
    tooltip=["Age", "Probability of using LinkedIn"]).properties(title="Probability of using LinkedIn by age").interactive()

st.altair_chart(age_plot, use_container_width=True)


###### 6 Dashboard for the marketing team

st.markdown("## Explore the survey data")
st.markdown("Check a box to show a chart. Each chart shows the share of survey respondents who use LinkedIn.")

if st.checkbox("LinkedIn use by income", key="plot1"):
    income_li = ss.groupby("income", as_index=False)[["sm_li"]].mean()
    income_df = pd.DataFrame({
        "Income level (1 = lowest, 9 = highest)": income_li["income"],
        "Share who use LinkedIn": income_li["sm_li"]
    })
    income_plot = alt.Chart(income_df).mark_bar().encode(
        x="Income level (1 = lowest, 9 = highest):N",
        y="Share who use LinkedIn",
        tooltip=["Income level (1 = lowest, 9 = highest)", "Share who use LinkedIn"]).properties(title="LinkedIn use rises with income")
    st.altair_chart(income_plot, use_container_width=True)

if st.checkbox("LinkedIn use by education", key="plot2"):
    education_li = ss.groupby("education", as_index=False)[["sm_li"]].mean()
    education_df = pd.DataFrame({
        "Education level (1 = less than high school, 8 = postgraduate)": education_li["education"],
        "Share who use LinkedIn": education_li["sm_li"]
    })
    education_plot = alt.Chart(education_df).mark_bar().encode(
        x="Education level (1 = less than high school, 8 = postgraduate):N",
        y="Share who use LinkedIn",
        tooltip=["Education level (1 = less than high school, 8 = postgraduate)", "Share who use LinkedIn"]).properties(title="LinkedIn use rises with education")
    st.altair_chart(education_plot, use_container_width=True)

if st.checkbox("LinkedIn use by age group", key="plot3"):
    age_groups = pd.DataFrame({
        "Age group": np.where(ss["age"] < 30, "18-29",
                     np.where(ss["age"] < 45, "30-44",
                     np.where(ss["age"] < 65, "45-64", "65+"))),
        "Share who use LinkedIn": ss["sm_li"]
    })
    age_group_li = age_groups.groupby("Age group", as_index=False)[["Share who use LinkedIn"]].mean()
    age_group_plot = alt.Chart(age_group_li).mark_bar().encode(
        x="Age group:N",
        y="Share who use LinkedIn",
        tooltip=["Age group", "Share who use LinkedIn"]).properties(title="LinkedIn use drops sharply at 65+")
    st.altair_chart(age_group_plot, use_container_width=True)

if st.checkbox("LinkedIn use by income and gender", key="plot4"):
    income_gender = ss.groupby(["income", "female"], as_index=False)[["sm_li"]].mean()
    income_gender_df = pd.DataFrame({
        "Income level (1 = lowest, 9 = highest)": income_gender["income"],
        "Share who use LinkedIn": income_gender["sm_li"],
        "Gender": np.where(income_gender["female"] == 1, "Female", "Male / other")
    })
    income_gender_plot = alt.Chart(income_gender_df).mark_circle(size=80).encode(
        x="Income level (1 = lowest, 9 = highest)",
        y="Share who use LinkedIn",
        color="Gender",
        tooltip=["Income level (1 = lowest, 9 = highest)", "Gender", "Share who use LinkedIn"]).properties(title="LinkedIn use by income and gender").interactive()
    st.altair_chart(income_gender_plot, use_container_width=True)

if st.checkbox("Model performance", key="plot5"):
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    st.markdown(f"**Accuracy:** {round(accuracy, 2)}   |   **Precision:** {round(precision, 2)}   |   **Recall:** {round(recall, 2)}")

    cm_df = pd.DataFrame(cm,
                         columns=["Predicted not a user", "Predicted user"],
                         index=["Actual not a user", "Actual user"])

    st.dataframe(cm_df.style.background_gradient(cmap="PiYG"))
