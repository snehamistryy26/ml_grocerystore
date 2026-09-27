import streamlit as st
import pandas as pd
import joblib
import os


MODEL_FILE = "train_model.pkl"
DATA_FILE = "Grocery_Sales_Retail_Analytics_24bcads054.csv"


st.set_page_config(
    page_title="Grocery Sales Retail Analytics",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Grocery Sales Retail Analytics")
st.header("Profit Prediction")


# Check model
if not os.path.exists(MODEL_FILE):
    st.error("train_model.pkl not found. First run: python train.py")
    st.stop()


# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()


# Load dataset for dropdown values
df = pd.read_csv(DATA_FILE)


st.write("Select order details to predict expected profit.")


col1, col2 = st.columns(2)


# -------------------------
# Dropdown inputs
# -------------------------

with col1:

    category = st.selectbox(
        "Category",
        sorted(df["Category"].dropna().unique())
    )

    sub_category = st.selectbox(
        "Sub Category",
        sorted(df["Sub Category"].dropna().unique())
    )

    city = st.selectbox(
        "City",
        sorted(df["City"].dropna().unique())
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].dropna().unique())
    )

    state = st.selectbox(
        "State",
        sorted(df["State"].dropna().unique())
    )


# -------------------------
# Numeric inputs
# -------------------------

with col2:

    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

    order_year = st.number_input(
        "Order Year",
        min_value=2000,
        max_value=2100,
        value=2017,
        step=1
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )

    order_day = st.number_input(
        "Order Day",
        min_value=1,
        max_value=31,
        value=1,
        step=1
    )


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Profit", type="primary"):

    input_data = pd.DataFrame([{
        "Category": category,
        "Sub Category": sub_category,
        "City": city,
        "Region": region,
        "State": state,
        "Sales": sales,
        "Discount": discount,
        "Order_Year": order_year,
        "Order_Month": order_month,
        "Order_Day": order_day
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Predicted Profit: ₹ {prediction:,.2f}"
    )


st.markdown("---")

st.caption(
    "Grocery Sales Retail Analytics | Machine Learning Project"
)