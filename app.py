# ---------------------------------------------------------
# Student Placement Prediction - Streamlit App
# (same steps as our notebook: clean -> encode -> split -> scale -> train)
# ---------------------------------------------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓", layout="wide")

# path of the csv file (it must be in the same folder as app.py)
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_placement_original_dataset.csv")


# ---------------------------------------------------------
# 1. Load and clean the data (same as notebook)
# ---------------------------------------------------------
@st.cache_data
def load_clean_data():
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    df = df.drop(['Student_ID', 'Salary_LPA'], axis=1)

    # text cleaning
    df['Extracurricular_Level'] = df['Extracurricular_Level'].fillna('none')
    obj_col = df.select_dtypes(exclude=np.number).columns.to_list()
    for i in obj_col:
        df[i] = df[i].str.strip().str.lower()

    df['Gender'] = df['Gender'].replace({'m': 'male', 'f': 'female', 'o': 'other'})
    df['College_Tier'] = df['College_Tier'].replace({'t1': 'tier 1', 't2': 'tier 2', 't3': 'tier 3',
                                                     'tier-1': 'tier 1', 'tier-2': 'tier 2', 'tier-3': 'tier 3'})
    df['Branch'] = df['Branch'].replace({
        'cse': 'computer science', 'comp science': 'computer science', 'computer science engineering': 'computer science',
        'it': 'information technology', 'info tech': 'information technology',
        'ai & ds': 'ai & data science', 'ai and data science': 'ai & data science', 'ai&ds': 'ai & data science', 'aids': 'ai & data science',
        'ece': 'electronics & communication', 'e&c': 'electronics & communication', 'electronics and communication': 'electronics & communication',
        'eee': 'electrical & electronics', 'electrical and electronics': 'electrical & electronics',
        'me': 'mechanical', 'mech': 'mechanical', 'mechanical engg': 'mechanical',
        'ce': 'civil', 'civil engg': 'civil',
        'chem': 'chemical', 'chemical engg': 'chemical'})

    # wrong numbers
    df.loc[df['SSLC_Percentage'] > 100, 'SSLC_Percentage'] = df['SSLC_Percentage'] / 10
    df.loc[df['CGPA'] > 10, 'CGPA'] = df['CGPA'] / 10
    df['Attendance_Percentage'] = df['Attendance_Percentage'].clip(upper=100)
    df['Aptitude_Score'] = df['Aptitude_Score'].abs().clip(upper=100)
    df['Current_Backlogs'] = df['Current_Backlogs'].abs()

    # missing values
    for i in df.columns:
        if not pd.api.types.is_numeric_dtype(df[i]):
            df[i] = df[i].fillna(df[i].mode()[0])
        else:
            df[i] = df[i].fillna(df[i].median())
    return df


# ---------------------------------------------------------
# 2. Encode, split, scale and train all 5 models
# ---------------------------------------------------------
@st.cache_resource
def train_models():
    df = load_clean_data().copy()

    # label encoding (we keep every encoder, so we can encode the user's input later)
    encoders = {}
    obj_col = df.select_dtypes(exclude=np.number).columns.to_list()
    for i in obj_col:
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
        encoders[i] = le

    x = df.drop('Placement_Status', axis=1)
    y = df['Placement_Status']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)

    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        rows.append([name, accuracy_score(ytest, ypred), precision_score(ytest, ypred),
                     recall_score(ytest, ypred), f1_score(ytest, ypred)])
    result = pd.DataFrame(rows, columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    result = result.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    return models, sc, encoders, list(x.columns), result


df = load_clean_data()
models, sc, encoders, feature_cols, result = train_models()

# ---------------------------------------------------------
# 3. Page layout
# ---------------------------------------------------------
st.title("🎓 Student Placement Prediction")
st.write("Enter a student's details and the machine learning model will predict whether the student will be **Placed** or **Not Placed**.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Insights", "🏆 Model Comparison"])

# ------------------ TAB 1 : PREDICTION ------------------
with tab1:
    algo = st.selectbox("Choose algorithm", list(models.keys()), index=0,
                        help="Logistic Regression gave the best accuracy in our project")

    st.subheader("Student details")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        branch = st.selectbox("Branch", encoders['Branch'].classes_)
        tier = st.selectbox("College Tier", encoders['College_Tier'].classes_)
        extra = st.selectbox("Extracurricular Level", encoders['Extracurricular_Level'].classes_)
        sslc = st.slider("SSLC Percentage", 45.0, 100.0, 75.0)
        hse = st.slider("HSE Percentage", 45.0, 100.0, 75.0)
    with c2:
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, step=0.01)
        attendance = st.slider("Attendance Percentage", 50.0, 100.0, 78.0)
        aptitude = st.slider("Aptitude Score", 0.0, 100.0, 55.0)
        coding = st.slider("Coding Skill Rating (1-10)", 1, 10, 5)
        comm = st.slider("Communication Score (1-10)", 1, 10, 5)
    with c3:
        backlogs = st.number_input("Current Backlogs", 0, 10, 0)
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 15, 2)
        certs = st.number_input("Certifications", 0, 15, 1)
        hackathons = st.number_input("Hackathons Participated", 0, 10, 0)
        mock = st.number_input("Mock Interviews Attended", 0, 10, 1)

    if st.button("Predict Placement", type="primary"):
        # put the input in a dictionary, encode the text values like in the notebook
        inp = {
            'Gender': encoders['Gender'].transform([gender])[0],
            'Branch': encoders['Branch'].transform([branch])[0],
            'College_Tier': encoders['College_Tier'].transform([tier])[0],
            'SSLC_Percentage': sslc, 'HSE_Percentage': hse, 'CGPA': cgpa,
            'Attendance_Percentage': attendance, 'Current_Backlogs': backlogs,
            'Internships': internships, 'Projects': projects, 'Certifications': certs,
            'Aptitude_Score': aptitude, 'Coding_Skill_Rating': coding, 'Communication_Score': comm,
            'Hackathons_Participated': hackathons,
            'Extracurricular_Level': encoders['Extracurricular_Level'].transform([extra])[0],
            'Mock_Interviews_Attended': mock,
        }
        xnew = pd.DataFrame([inp])[feature_cols]   # same column order as training
        xnew = sc.transform(xnew)                  # same scaling as training

        model = models[algo]
        pred = model.predict(xnew)[0]
        prob = model.predict_proba(xnew)[0]        # [prob not placed, prob placed]
        # encoder classes are alphabetical: 'not placed' = 0, 'placed' = 1
        placed_chance = prob[1] * 100

        st.divider()
        if pred == 1:
            st.success(f"✅ The student is likely to be **PLACED**  (chance: {placed_chance:.1f}%)")
        else:
            st.error(f"❌ The student is likely to be **NOT PLACED**  (chance of placement: {placed_chance:.1f}%)")
        st.progress(int(placed_chance))
        st.caption(f"Prediction made using **{algo}**. This is only a prediction based on past data, not a guarantee.")

# ------------------ TAB 2 : VISUALIZATIONS ------------------
with tab2:
    st.subheader("Understanding the data")
    st.write(f"The cleaned dataset has **{df.shape[0]} students** and **{df.shape[1]} columns**.")
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown("**1. Placed vs Not Placed**")
        fig, ax = plt.subplots()
        sns.countplot(x='Placement_Status', data=df, ax=ax)
        st.pyplot(fig)
        st.caption("More students are placed than not placed.")
    with v2:
        st.markdown("**2. CGPA vs Placement**")
        fig, ax = plt.subplots()
        sns.boxplot(x='Placement_Status', y='CGPA', data=df, ax=ax)
        st.pyplot(fig)
        st.caption("Placed students have a higher CGPA.")
    with v3:
        st.markdown("**3. Internships vs Placement**")
        fig, ax = plt.subplots()
        sns.countplot(x='Internships', hue='Placement_Status', data=df, ax=ax)
        st.pyplot(fig)
        st.caption("More internships means more chance of placement.")

    with st.expander("Show cleaned dataset"):
        st.dataframe(df.head(100))

# ------------------ TAB 3 : MODEL COMPARISON ------------------
with tab3:
    st.subheader("Performance of all 5 algorithms")
    st.dataframe(result.style.format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1 Score': '{:.3f}'}))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Algorithm', y='Accuracy', data=result, ax=ax)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20)
    st.pyplot(fig)
    best = result.iloc[0]
    st.info(f"🏆 Best algorithm: **{best['Algorithm']}** with accuracy **{best['Accuracy']*100:.1f}%**")
