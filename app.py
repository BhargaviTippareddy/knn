import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import learning_curve

# Streamlit App
st.set_page_config(page_title="Decision Boundary & Learning Curve", page_icon="📊")

st.title("Decision Boundary & Learning Curve Visualization")
st.image(r"innomatics.jpg")
st.write("Choose a classifier to visualize the decision boundary and learning curve.")

# Generate dataset
X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, class_sep=1, random_state=27)

# Classifier selection
classifier_name = st.selectbox("Select Classifier", ("KNN", "Decision Tree", "Logistic Regression"))

# Train model based on selection
if classifier_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif classifier_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif classifier_name == "Logistic Regression":
    model = LogisticRegression()

# Fit the model
model.fit(X, y)

# Plot decision boundary
st.write(f"### Decision Boundary for {classifier_name}")
fig, ax = plt.subplots()
plot_decision_regions(X, y, clf=model, legend=2)
st.pyplot(fig)

# Function to plot learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title(f"Learning Curve for {classifier_name}")
    plt.legend(loc="best")
    st.pyplot(plt)

# Plot learning curve
st.write(f"### Learning Curve for {classifier_name}")
plot_learning_curve(model, X, y)
