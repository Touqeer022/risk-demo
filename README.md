# 🏦 SkyBank AI - Credit Scoring & Loan Approval System

**SkyBank AI** is a machine learning-based application designed to automate the loan approval process. It analyzes customer demographic and financial data to predict credit risk and generates instant loan offers for eligible applicants.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)

## 🚀 Key Features

- **🤖 AI-Powered Risk Assessment:** Uses a Random Forest Classifier to predict if a customer is "Good" or "Bad" (High Risk).
- **📊 Interactive Dashboard:** User-friendly interface for bank officers to input details.
- **📄 Automated Loan Offers:** Instantly generates EMI, Interest Rate, and Tenure details for approved loans.
- **🛡️ Real-time Validation:** Checks inputs against the trained model logic.

## 📂 Dataset Used

The model is trained on the **German Credit Risk Dataset**, a standard benchmark for credit scoring algorithms. It considers factors like:
- Age & Gender
- Job Skill Level
- Housing Status (Own/Rent)
- Saving & Checking Account Balance
- Loan Amount & Duration

## 🛠️ Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/credit-scoring-project.git](https://github.com/YOUR-USERNAME/credit-scoring-project.git)
cd credit-scoring-project
2. Install Dependencies
Make sure you have Python installed. Then run:

Bash
pip install pandas scikit-learn joblib streamlit numpy
3. Train the Model (Important!)
Before running the app, generate the AI model (.pkl files) by running the training script:

Bash
python train_model.py
You will see a success message: "🎉 Success! 'credit_model.pkl' file created."

4. Run the Application
Launch the dashboard using Streamlit:

Bash
streamlit run app.py
(If you face a command error, try: python -m streamlit run app.py)

🧠 How It Works
Data Processing: The system takes user inputs (Age, Job, Savings, etc.).

Encoding: Converts text data into machine-readable numbers using saved Encoders.

Prediction: The trained RandomForest model calculates the probability of default.

Decision:

Low Risk: Displays "Approved" with a custom loan offer.

High Risk: Displays "Rejected" with reasoning.

📂 Project Structure
app.py: Main application code (Streamlit Dashboard).

train_model.py: Script to preprocess data and train the ML model.

german_credit_data.csv: The dataset used for training.

credit_model.pkl: The saved AI model (generated after training).

encoders.pkl: Saved label encoders for data transformation.

🤝 Contributing
Feel free to fork this project and submit pull requests.

Developed for AI Project Showcase (Idea #20)
