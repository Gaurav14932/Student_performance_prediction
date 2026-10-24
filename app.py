from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = Path(__file__).with_name("AI-Data.csv")
NUMERIC_FEATURES = ["raisedhands", "VisITedResources", "Discussion"]
CATEGORICAL_FEATURES = ["StudentAbsenceDays"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Class"
CHART_COLUMNS = [
    "gender",
    "NationalITy",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]


st.set_page_config(page_title="Student Performance Prediction", page_icon="📚")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {', '.join(missing_columns)}")

    return data


def make_pipeline(classifier):
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("absence", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


@st.cache_resource
def train_models(data):
    features = data[FEATURES]
    target = data[TARGET]
    class_labels = sorted(target.unique())
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Perceptron": Perceptron(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network (MLP)": MLPClassifier(
            activation="logistic",
            max_iter=1000,
            random_state=42,
        ),
    }

    models = {}
    accuracies = {}
    reports = {}

    for name, classifier in classifiers.items():
        model = make_pipeline(classifier)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        models[name] = model
        accuracies[name] = accuracy_score(y_test, predictions)
        reports[name] = classification_report(
            y_test,
            predictions,
            labels=class_labels,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )

    return models, accuracies, reports, x_test, y_test, class_labels


st.title("Student Performance Prediction")
st.write("Explore the student dataset, compare classifiers, and try a sample prediction.")

data = load_data()
models, accuracies, reports, x_test, y_test, class_labels = train_models(data)

page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Explore the data", "Model performance", "Try a prediction"],
)

if page == "Overview":
    st.subheader("Dataset overview")
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Students in dataset", len(data))
    second_metric.metric("Dataset columns", len(data.columns))
    third_metric.metric("Performance classes", data[TARGET].nunique())
    st.write("A preview of the dataset:")
    st.dataframe(data.head(10), use_container_width=True)

elif page == "Explore the data":
    st.subheader("Explore the data")
    chart_column = st.selectbox("Compare performance by", CHART_COLUMNS)

    figure, axis = plt.subplots(figsize=(10, 5))
    sns.countplot(data=data, x=chart_column, hue=TARGET, hue_order=class_labels, ax=axis)
    axis.set_title(f"Student performance by {chart_column}")
    axis.set_xlabel(chart_column)
    axis.set_ylabel("Number of students")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Numeric feature correlations")
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=axis)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

elif page == "Model performance":
    st.subheader("Compare classifiers")
    accuracy_data = pd.DataFrame(
        [{"Classifier": name, "Accuracy": accuracy} for name, accuracy in accuracies.items()]
    ).sort_values("Accuracy", ascending=False)
    st.bar_chart(accuracy_data, x="Classifier", y="Accuracy")

    selected_model = st.selectbox("Show detailed results for", list(models))
    predictions = models[selected_model].predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)

    st.write(f"**Test accuracy:** {accuracies[selected_model]:.1%}")
    st.write("Rows are actual classes; columns are predicted classes.")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in class_labels],
            columns=[f"Predicted {label}" for label in class_labels],
        ),
        use_container_width=True,
    )
    st.write("Classification report:")
    st.dataframe(pd.DataFrame(reports[selected_model]).transpose(), use_container_width=True)

elif page == "Try a prediction":
    st.subheader("Try a sample prediction")
    st.caption("Enter values for the three numeric features and choose an absence category.")

    input_values = {}
    for feature in NUMERIC_FEATURES:
        minimum = int(data[feature].min())
        maximum = int(data[feature].max())
        input_values[feature] = st.slider(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=int(data[feature].median()),
        )

    input_values["StudentAbsenceDays"] = st.selectbox(
        "StudentAbsenceDays",
        sorted(data["StudentAbsenceDays"].dropna().unique()),
    )
    selected_model = st.selectbox("Classifier", list(models))

    if st.button("Predict performance"):
        sample = pd.DataFrame([input_values], columns=FEATURES)
        prediction = models[selected_model].predict(sample)[0]
        st.success(f"Predicted performance class: {prediction}")
        st.caption("This is a model prediction based on the project dataset, not a guarantee.")
