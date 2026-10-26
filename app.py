
# ==========================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Built with Streamlit
# ==========================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "Enter a customer's information below to estimate "
    "their probability of leaving the service."
)

# ==========================================
# 2. LOAD SAVED MODEL FILES
# ==========================================

APP_FOLDER = Path(__file__).resolve().parent

try:
    log_model = joblib.load(APP_FOLDER / "tel_churn.pkl")
    scaler = joblib.load(APP_FOLDER / "tel_scaler.pkl")
    features = joblib.load(APP_FOLDER / "tel_features.pkl")

except Exception as e:
    st.error(f"Unable to load the model files: {e}")
    st.info(
        "Ensure tel_churn.pkl, tel_scaler.pkl and "
        "tel_features.pkl are in the same folder as app.py."
    )
    st.stop()


# ==========================================
# 3. DEFINE COLUMNS
# ==========================================

num_cols = [
    "monthly_charges",
    "total_charges",
    "tenure_months",
    "satisfaction_score",
    "num_support_tickets",
    "avg_monthly_value",
    "charges_per_ticket",
    "engagement_score"
]

categorical_cols = [
    "contract_type",
    "internet_service",
    "payment_method"
]


# Get the original category names from the saved
# feature list to ensure the app uses the right labels.

def get_categories(column_name):
    prefix = column_name + "_"

    categories = [
        col[len(prefix):]
        for col in features
        if col.startswith(prefix)
    ]

    if not categories:
        raise ValueError(
            f"No encoded categories found for {column_name}. "
            "Check tel_features.pkl."
        )

    return categories


try:
    contract_options = get_categories("contract_type")
    internet_options = get_categories("internet_service")
    payment_options = get_categories("payment_method")

except ValueError as e:
    st.error(str(e))
    st.stop()


# ==========================================
# 4. CUSTOMER INPUT FORM
# ==========================================

with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure_months = st.number_input(
            "Tenure (months)",
            min_value=1,
            max_value=600,
            value=12,
            step=1
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.01,
            value=70.0,
            step=5.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

        satisfaction_score = st.number_input(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

        num_support_tickets = st.number_input(
            "Number of Support Tickets",
            min_value=0,
            max_value=1000,
            value=1,
            step=1
        )

    with col2:
        contract_type = st.selectbox(
            "Contract Type",
            options=contract_options
        )

        internet_service = st.selectbox(
            "Internet Service",
            options=internet_options
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=payment_options
        )

        tech_support = st.selectbox(
            "Has Tech Support?",
            options=["Yes", "No"]
        )

        streaming_service = st.selectbox(
            "Uses Streaming Service?",
            options=["Yes", "No"]
        )

        paperless_billing = st.selectbox(
            "Uses Paperless Billing?",
            options=["Yes", "No"]
        )

    st.divider()

    st.subheader("Prediction Settings")

    threshold = st.slider(
        "Churn Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Lower thresholds generally identify more potential "
            "churners but may produce more false alarms."
        )
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 5. PROCESS CUSTOMER INPUT
# ==========================================

if submitted:

    # Convert Yes/No answers into binary values.
    # This assumes the original dataset uses 1 for Yes
    # and 0 for No in these three columns.

    tech_support_value = 1 if tech_support == "Yes" else 0
    streaming_value = 1 if streaming_service == "Yes" else 0
    paperless_value = 1 if paperless_billing == "Yes" else 0

    # Create the customer record.

    customer = {
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "satisfaction_score": satisfaction_score,
        "num_support_tickets": num_support_tickets,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support_value,
        "streaming_service": streaming_value,
        "paperless_billing": paperless_value
    }

    input_df = pd.DataFrame([customer])

    # ==========================================
    # 6. FEATURE ENGINEERING
    # ==========================================

    input_df["avg_monthly_value"] = (
        input_df["total_charges"] /
        input_df["tenure_months"]
    )

    input_df["charges_per_ticket"] = (
        input_df["monthly_charges"] /
        (input_df["num_support_tickets"] + 1)
    )

    input_df["engagement_score"] = (
        input_df["tenure_months"] * 0.4
        + input_df["satisfaction_score"] * 0.3
        - input_df["num_support_tickets"] * 0.3
    )

    input_df["high_risk"] = (
        (input_df["satisfaction_score"] <= 2)
        & (input_df["num_support_tickets"] >= 3)
    ).astype(int)

    # Use the same tenure groups as the training notebook.

    def tenure_group(tenure):
        if tenure <= 6:
            return "new"
        elif tenure <= 36:
            return "mid"
        else:
            return "old"

    input_df["tenure_group"] = (
        input_df["tenure_months"].apply(tenure_group)
    )

    # ==========================================
    # 7. ENCODE CATEGORICAL COLUMNS
    # ==========================================

    input_df = pd.get_dummies(
        input_df,
        columns=[
            "contract_type",
            "internet_service",
            "payment_method",
            "tenure_group"
        ],
        dtype=int
    )

    # Make the columns match the training features exactly.
    # Missing dummy columns are filled with zero.

    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    )

    # ==========================================
    # 8. SCALE NUMERICAL COLUMNS
    # ==========================================

    try:
        # Use the already-fitted scaler.
        # Do not fit a new scaler in the app.

        input_df[num_cols] = scaler.transform(
            input_df[num_cols].astype(float)
        )

        # ==========================================
        # 9. MAKE PREDICTION
        # ==========================================

        probabilities = log_model.predict_proba(input_df)[0]

        # Find the probability belonging to churn class 1.

        churn_class_index = list(log_model.classes_).index(1)
        churn_probability = probabilities[churn_class_index]

        # Apply the selected classification threshold.

        prediction = int(churn_probability >= threshold)

        # ==========================================
        # 10. DISPLAY RESULTS
        # ==========================================

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.1%}"
            )

        with result_col2:
            st.metric(
                "Selected Threshold",
                f"{threshold:.0%}"
            )

        if prediction == 1:
            st.error(
                "⚠️ **Potential Churner**\n\n"
                "This customer is classified as likely to churn "
                "at the selected threshold."
            )

            st.write(
                "Recommended action: Review the customer's "
                "satisfaction, support history, contract and "
                "service experience. Consider appropriate "
                "retention measures."
            )

        else:
            st.success(
                "✅ **Not Classified as a Churner**\n\n"
                "The estimated churn probability is below "
                "the selected threshold."
            )

            st.write(
                "Continue monitoring the customer's experience "
                "and service engagement."
            )

        st.caption(
            "This is a model-based estimate, not a guarantee "
            "of future customer behaviour. Lowering the "
            "threshold may capture more churners but can "
            "also increase false positives."
        )

    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")

        st.info(
            "Check that the model, scaler and feature list "
            "were saved from the same training workflow, "
            "and that the numerical columns match."
        )
