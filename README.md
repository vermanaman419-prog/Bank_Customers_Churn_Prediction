
# 🏦 Bank Customer Churn Prediction
## 📊 Predicting Customer Attrition with Machine Learning

This project builds a Customer Churn Prediction Model using structured banking data to identify customers who are likely to leave the bank.
The workflow includes data preprocessing, model training (Logistic Regression), evaluation, and deployment via Gradio + HuggingFace/Render.

# 🚀 Project Overview

**Objective**: Predict whether a customer is likely to churn based on demographic and behavioral features.

**Goal**: Enable banks to take proactive actions to improve retention and reduce revenue loss.

# 🧰 Tools & Technologies
- Category	Tools Used
- Programming	- Python (Pandas, NumPy, Scikit-learn)
- ML Model	- Logistic Regression
- Visualization	- Matplotlib, Seaborn
- Deployment	- Gradio, HuggingFace Spaces / Render
- Environment -	Google Colab
- Model Storage -	Joblib

# 🔍 Key Steps

- Data Cleaning & Preprocessing

- Removed inconsistencies and handled missing values.

- Selected 7 numerical features for modeling:

- CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember

- Exploratory Data Analysis (EDA)

- Examined churn distribution

- Identified relationships between features and churn probability

- Visualized trends with heatmaps, count plots, and pairwise relationships

- Model Training (Logistic Regression)

- Split data into training and test sets

- Trained a baseline logistic regression classifier

- Evaluated using accuracy, precision, recall, F1-score, ROC curve

- Deployment (Gradio App)

- Built a simple user interface to input customer features

- Displayed churn prediction results along with probability

- Hosted app on HuggingFace/Render

# 📈 Insights

- Customers with low balance, low tenure, and higher age showed higher chances of churn.

- Inactive members and customers who do not hold credit cards were more likely to leave.

- Logistic Regression provided a clear, interpretable baseline model for churn classification.

🤖 Live Model Demo

🔗 Gradio App: https://naman419-churn-prediction.hf.space/?__theme=system&deep_link=1GEJCPuT73c

# 🧠 Learnings

- Improved understanding of binary classification and evaluation metrics.

- Learned how to train and test Logistic Regression on real-world data.

- Strengthened concepts of model deployment using Gradio & cloud platforms.

- Enhanced the ability to convert raw customer data into actionable business insights.

# 👨‍💻 Author

Naman Verma
📍 Gurugram, India
📧 vermanaman419@gmail.com
🔗 LinkedIn : https://www.linkedin.com/in/naman419/

