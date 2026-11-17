# Who's a LinkedIn User?

A Streamlit app that predicts whether someone uses LinkedIn from six demographic traits, and shows how likely they are to be a user.

**Live app:** [add Streamlit Cloud link]

Built for Programming II in Georgetown's MSBA program. The brief: a marketing team wants to know which audiences are worth reaching on LinkedIn, so the app had to make predictions in real time from user input.

---

## What it does

Enter a person's details and the app returns:

- **A prediction:** LinkedIn user or not
- **A probability:** how likely they are to use LinkedIn
- [add any charts or extras you included, e.g. how the probability changes with age]

## The model

- **Data:** Pew Research Center survey on social media use among US adults
- **Features:** income (1–9), education (1–8), parent, married, female, age
- **Target:** whether the person uses LinkedIn (1 = yes, 0 = no)
- **Model:** logistic regression with `class_weight="balanced"`, since LinkedIn users are the smaller group
- **Split:** 80% training, 20% test
- **Results on the test set:** accuracy [__], precision [__], recall [__], F1 [__]

One finding: holding everything else equal, a 42-year-old and an 82-year-old with high income and education get very different predictions. [Add your two probabilities here.]

## Run it locally

```bash
git clone https://github.com/rpradaa/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `social_media_usage.csv` | Survey data |
| `requirements.txt` | Packages needed to run the app |
| `[notebook].ipynb` | Data cleaning, analysis and model building |

## Data note

The data come from Pew Research Center. They are publicly available and used here for educational purposes only.
