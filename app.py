import streamlit as st
import pandas as pd
import joblib

# --- 1. PAGE CONFIGURATION (Isse look change hoga) ---
st.set_page_config(
    page_title="SkyBank Loan System",
    page_icon="🏦",
    layout="wide"  # Poori screen use karega (Dashboard look)
)

# --- 2. MODEL LOAD KAREIN ---
try:
    model = joblib.load('credit_model.pkl')
    encoders = joblib.load('encoders.pkl')
except:
    st.error("⚠️ Model files missing! Please run 'python train_model.py' first.")
    st.stop()

# --- 3. CUSTOM CSS (Thoda styling taake alag dikhe) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        height: 50px;
        font-size: 18px;
        border-radius: 10px;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR INPUTS (Saara form left side par) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Loan Application")
    st.write("Fill details to check eligibility.")
    
    # Customer Name (Sirf display ke liye)
    cust_name = st.text_input("Customer Name", "John Doe")
    
    st.divider()
    
    # Real Inputs
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["male", "female"])
    job = st.selectbox("Job Level", [0, 1, 2, 3], help="0=Unskilled, 3=Highly Skilled")
    housing = st.selectbox("Housing Type", ["own", "rent", "free"])
    
    st.subheader("Financials")
    credit_amount = st.number_input("Loan Amount ($)", 0, 20000, 2000)
    duration = st.slider("Duration (Months)", 1, 72, 12)
    saving = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich"])
    checking = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "repairs", "vacation/others"])
    
    # Bada Button
    st.write("")
    run_btn = st.button("🚀 Analyze Risk")

# --- 5. MAIN SCREEN (Right side) ---
st.title("🏦 SkyBank AI Credit System")
st.write("---")

if run_btn:
    # Data Preparation
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [gender],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving],
        'Checking account': [checking],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })

    # Encoding (Text to Number)
    for col, le in encoders.items():
        if col != 'Risk':
            input_data[col] = input_data[col].astype(str)
            input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_data[col] = le.transform(input_data[col])

    # Prediction
    prediction = model.predict(input_data)[0]
    risk_status = encoders['Risk'].inverse_transform([prediction])[0]

    # --- 6. DISPLAY RESULTS (Naya Style) ---
    col_res1, col_res2 = st.columns([2, 1])

    with col_res1:
        if risk_status == 'good':
            # --- APPROVED UI ---
            st.balloons() # Thoda celebration style
            st.markdown(f"""
                <div class="success-box">
                    <h2>✅ LOAN APPROVED</h2>
                    <p>Congratulations <b>{cust_name}</b>! You are eligible for this loan.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.subheader("📄 Loan Offer Details")
            
            # Data ko Table format mein dikhana (Aapki Requirement)
            details = {
                "Applicant Name": [cust_name],
                "Approved Amount": [f"${credit_amount}"],
                "Duration": [f"{duration} Months"],
                "Interest Rate": ["5.4% (Fixed)"],  # Fake static value for realism
                "Monthly EMI": [f"${round(credit_amount/duration * 1.1, 2)}"] # Approx calculation
            }
            st.table(pd.DataFrame(details).T.rename(columns={0: 'Details'}))

        else:
            # --- REJECTED UI ---
            st.error("⛔ LOAN REJECTED")
            st.write(f"Sorry **{cust_name}**, based on the AI analysis, this application carries high risk.")
            st.warning("Reason: The loan amount might be too high compared to the duration and financial history.")

    with col_res2:
        # Side Metrics
        st.metric("Risk Score", "Low" if risk_status == 'good' else "High", delta="Safe" if risk_status=='good' else "-Critical")
        st.metric("Credit Limit", "$20,000")
        st.info("💡 Tip: Try reducing the loan amount for better approval chances.")

else:
    # Jab tak button nahi dabaya, yeh dikhega
    st.info("👈 Please fill the details in the sidebar to process the loan.")
    st.image("https://img.freepik.com/free-vector/bank-loan-concept-illustration_114360-18437.jpg", width=500)
