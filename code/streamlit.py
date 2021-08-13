# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

pyo.init_notebook_mode()
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
import streamlit as st
from imblearn.over_sampling import SMOTE


# %%
c_data = pd.read_csv("./assets/BankChurners.csv")
c_data = c_data[c_data.columns[:-2]]
# c_data.head(3)
c_data.shape


# %%
st.write("When we look at our data set, we see that there about ten thousands samples")


# %%
st.title("Exploratory data analysis")


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data["Customer_Age"], name="Age Box Plot", boxmean=True)
tr2 = go.Histogram(x=c_data["Customer_Age"], name="Age Histogram")

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(title_text="Distribution of Customer Ages")
fig.show()
st.write(fig)


# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "",
        "<b>Platinum Card Holders",
        "<b>Blue Card Holders<b>",
        "Residuals",
    ),
    vertical_spacing=0.09,
    specs=[
        [{"type": "pie", "rowspan": 2}, {"type": "pie"}],
        [None, {"type": "pie"}],
    ],
)

fig.add_trace(
    go.Pie(
        values=c_data.Gender.value_counts().values,
        labels=["<b>Female<b>", "<b>Male<b>"],
        hole=0.3,
        pull=[0, 0.3],
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Pie(
        labels=["Female Platinum Card Holders", "Male Platinum Card Holders"],
        values=c_data.query('Card_Category=="Platinum"').Gender.value_counts().values,
        pull=[0, 0.05, 0.5],
        hole=0.3,
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Pie(
        labels=["Female Blue Card Holders", "Male Blue Card Holders"],
        values=c_data.query('Card_Category=="Blue"').Gender.value_counts().values,
        pull=[0, 0.2, 0.5],
        hole=0.3,
    ),
    row=2,
    col=2,
)


fig.update_layout(
    showlegend=True,
    title_text="<b>Distribution Of Gender And Different Card Statuses<b>",
)

fig.show()
st.write(fig)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data["Dependent_count"], name="Dependent count Box Plot", boxmean=True)
tr2 = go.Histogram(x=c_data["Dependent_count"], name="Dependent count Histogram")

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(title_text="Distribution of Dependent counts (close family size)")
fig.show()
st.write(fig)


# %%
ex.pie(
    c_data, names="Education_Level", title="Propotion Of Education Levels", hole=0.33
)


# %%
ex.pie(
    c_data,
    names="Marital_Status",
    title="Propotion Of Different Marriage Statuses",
    hole=0.33,
)


# %%
ex.pie(
    c_data,
    names="Income_Category",
    title="Propotion Of Different Income Levels",
    hole=0.33,
)


# %%
ex.pie(
    c_data,
    names="Card_Category",
    title="Propotion Of Different Card Categories",
    hole=0.33,
)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data["Months_on_book"], name="Months on book Box Plot", boxmean=True)
tr2 = go.Histogram(x=c_data["Months_on_book"], name="Months on book Histogram")

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(title_text="Distribution of months the customer is part of the bank")
fig.show()
st.write(fig)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(
    x=c_data["Total_Relationship_Count"],
    name="Total no. of products Box Plot",
    boxmean=True,
)
tr2 = go.Histogram(
    x=c_data["Total_Relationship_Count"], name="Total no. of products Histogram"
)

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(
    title_text="Distribution of Total no. of products held by the customer"
)
fig.show()
st.write(fig)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(
    x=c_data["Months_Inactive_12_mon"],
    name="number of months inactive Box Plot",
    boxmean=True,
)
tr2 = go.Histogram(
    x=c_data["Months_Inactive_12_mon"], name="number of months inactive Histogram"
)

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(
    title_text="Distribution of the number of months inactive in the last 12 months"
)
fig.show()
st.write(fig)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data["Credit_Limit"], name="Credit_Limit Box Plot", boxmean=True)
tr2 = go.Histogram(x=c_data["Credit_Limit"], name="Credit_Limit Histogram")

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(title_text="Distribution of the Credit Limit")
fig.show()
st.write(fig)


# %%
fig = make_subplots(rows=2, cols=1)

tr1 = go.Box(x=c_data["Total_Trans_Amt"], name="Total_Trans_Amt Box Plot", boxmean=True)
tr2 = go.Histogram(x=c_data["Total_Trans_Amt"], name="Total_Trans_Amt Histogram")

fig.add_trace(tr1, row=1, col=1)
fig.add_trace(tr2, row=2, col=1)

fig.update_layout(
    title_text="Distribution of the Total Transaction Amount (Last 12 months)"
)
fig.show()
st.write(fig)


# %%
ex.pie(
    c_data,
    names="Attrition_Flag",
    title="Proportion of churn vs not churn customers",
    hole=0.33,
)


# %%
# data preprocessing
c_data.Attrition_Flag = c_data.Attrition_Flag.replace(
    {"Attrited Customer": 1, "Existing Customer": 0}
)
c_data.Gender = c_data.Gender.replace({"F": 1, "M": 0})
c_data = pd.concat(
    [c_data, pd.get_dummies(c_data["Education_Level"]).drop(columns=["Unknown"])],
    axis=1,
)
c_data = pd.concat(
    [c_data, pd.get_dummies(c_data["Income_Category"]).drop(columns=["Unknown"])],
    axis=1,
)
c_data = pd.concat(
    [c_data, pd.get_dummies(c_data["Marital_Status"]).drop(columns=["Unknown"])], axis=1
)
c_data = pd.concat(
    [c_data, pd.get_dummies(c_data["Card_Category"]).drop(columns=["Platinum"])], axis=1
)
c_data.drop(
    columns=[
        "Education_Level",
        "Income_Category",
        "Marital_Status",
        "Card_Category",
        "CLIENTNUM",
    ],
    inplace=True,
)


# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Perason Correaltion", "Spearman Correaltion"),
)
colorscale = [
    [1.0, "rgb(165,0,38)"],
    [0.8888888888888888, "rgb(215,48,39)"],
    [0.7777777777777778, "rgb(244,109,67)"],
    [0.6666666666666666, "rgb(253,174,97)"],
    [0.5555555555555556, "rgb(254,224,144)"],
    [0.4444444444444444, "rgb(224,243,248)"],
    [0.3333333333333333, "rgb(171,217,233)"],
    [0.2222222222222222, "rgb(116,173,209)"],
    [0.1111111111111111, "rgb(69,117,180)"],
    [0.0, "rgb(49,54,149)"],
]

s_val = c_data.corr("pearson")
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(
        x=s_col,
        y=s_idx,
        z=s_val,
        name="pearson",
        showscale=False,
        xgap=0.7,
        ygap=0.7,
        colorscale=colorscale,
    ),
    row=1,
    col=1,
)


s_val = c_data.corr("spearman")
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col, y=s_idx, z=s_val, xgap=0.7, ygap=0.7, colorscale=colorscale),
    row=2,
    col=1,
)
fig.update_layout(
    height=700, hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
)
fig.update_layout(title_text="Numeric Correaltions")
fig.show()
st.write(fig)


# %%
oversample = SMOTE()
X, y = oversample.fit_resample(c_data[c_data.columns[1:]], c_data[c_data.columns[0]])
usampled_df = X.assign(Churn=y)
upsampled_untouched_df = usampled_df.copy()


# %%
ohe_data = usampled_df[usampled_df.columns[15:-1]].copy()
usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])


# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Perason Correaltion", "Spearman Correaltion"),
)
colorscale = [
    [1.0, "rgb(165,0,38)"],
    [0.8888888888888888, "rgb(215,48,39)"],
    [0.7777777777777778, "rgb(244,109,67)"],
    [0.6666666666666666, "rgb(253,174,97)"],
    [0.5555555555555556, "rgb(254,224,144)"],
    [0.4444444444444444, "rgb(224,243,248)"],
    [0.3333333333333333, "rgb(171,217,233)"],
    [0.2222222222222222, "rgb(116,173,209)"],
    [0.1111111111111111, "rgb(69,117,180)"],
    [0.0, "rgb(49,54,149)"],
]

s_val = upsampled_untouched_df.corr("pearson")
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(
        x=s_col,
        y=s_idx,
        z=s_val,
        name="pearson",
        showscale=False,
        xgap=1,
        ygap=1,
        colorscale=colorscale,
    ),
    row=1,
    col=1,
)


s_val = upsampled_untouched_df.corr("spearman")
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values
fig.add_trace(
    go.Heatmap(x=s_col, y=s_idx, z=s_val, xgap=1, ygap=1, colorscale=colorscale),
    row=2,
    col=1,
)
fig.update_layout(
    height=1000, hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
)
fig.update_layout(title_text="Upsmapled Correlations with all the features included")
fig.show()
st.write(fig)


# %%
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42)


# %%
rf_pipe = Pipeline(
    steps=[("scale", StandardScaler()), ("RF", RandomForestClassifier(random_state=42))]
)
ada_pipe = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("RF", AdaBoostClassifier(random_state=42, learning_rate=0.7)),
    ]
)
svm_pipe = Pipeline(
    steps=[("scale", StandardScaler()), ("RF", SVC(random_state=42, kernel="rbf"))]
)


f1_cross_val_scores = cross_val_score(rf_pipe, train_x, train_y, cv=5, scoring="f1")
ada_f1_cross_val_scores = cross_val_score(
    ada_pipe, train_x, train_y, cv=5, scoring="f1"
)
svm_f1_cross_val_scores = cross_val_score(
    svm_pipe, train_x, train_y, cv=5, scoring="f1"
)


# %%
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Random Forest Cross Val Scores",
        "Adaboost Cross Val Scores",
        "SVM Cross Val Scores",
    ),
)

fig.add_trace(
    go.Scatter(
        x=list(range(0, len(f1_cross_val_scores))),
        y=f1_cross_val_scores,
        name="Random Forest",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, len(ada_f1_cross_val_scores))),
        y=ada_f1_cross_val_scores,
        name="Adaboost",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, len(svm_f1_cross_val_scores))),
        y=svm_f1_cross_val_scores,
        name="SVM",
    ),
    row=3,
    col=1,
)

fig.update_layout(title_text="Different Model 5 Fold Cross Validation")
fig.update_yaxes(title_text="F1 Score")
fig.update_xaxes(title_text="Fold #")

fig.show()
st.write(fig)


# %%
st.title("Model evaluation")


# %%
rf_pipe.fit(train_x, train_y)
rf_prediction = rf_pipe.predict(test_x)

ada_pipe.fit(train_x, train_y)
ada_prediction = ada_pipe.predict(test_x)

svm_pipe.fit(train_x, train_y)
svm_prediction = svm_pipe.predict(test_x)


# %%
fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=["<b>Model<b>", "<b>F1 Score On Test Data<b>"],
                line_color="darkslategray",
                fill_color="whitesmoke",
                align=["center", "center"],
                font=dict(color="black", size=18),
                height=40,
            ),
            cells=dict(
                values=[
                    ["<b>Random Forest<b>", "<b>AdaBoost<b>", "<b>SVM<b>"],
                    [
                        np.round(f1(rf_prediction, test_y), 2),
                        np.round(f1(ada_prediction, test_y), 2),
                        np.round(f1(svm_prediction, test_y), 2),
                    ],
                ],
                font=dict(color="black", size=16)
            ),
        )
    ]
)

fig.update_layout(title="Model Results On Test Data")
fig.show()
st.write(fig)


# %%
unsampled_data_prediction_RF = rf_pipe.predict(c_data[c_data.columns[1:]])
unsampled_data_prediction_ADA = ada_pipe.predict(c_data[c_data.columns[1:]])
unsampled_data_prediction_SVM = svm_pipe.predict(c_data[c_data.columns[1:]])


# %%
z = confusion_matrix(unsampled_data_prediction_RF, c_data[c_data.columns[0]])
fig = ff.create_annotated_heatmap(
    z,
    x=["Not Churn", "Churn"],
    y=["Predicted Not Churn", "Predicted Churn"],
    colorscale="Fall",
    xgap=3,
    ygap=3,
)
fig["data"][0]["showscale"] = True
fig.update_layout(
    title="Prediction On Original Data With Random Forest Model Confusion Matrix"
)
fig.show()
st.write(fig)
