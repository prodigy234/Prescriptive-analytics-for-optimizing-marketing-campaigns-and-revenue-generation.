# 📦 Amazon Product Prescriptive Analytics Dashboard

This interactive Streamlit dashboard provides comprehensive **descriptive, diagnostic, and prescriptive analytics** for Amazon product data, allowing users to filter by product category, price, and rating, while exploring visual insights and actionable recommendations.

---

This highly intelligent project which is a powerful Streamlit-based Prescriptive Analytics Dashboard which I built using real Amazon product data can be accessed live on streamlit [Here](https://amazonprescriptivedashboard.streamlit.app/)

---

This interactive Streamlit dashboard provides:

**📊 Interactive Visualizations (using Plotly and Seaborn)**

**🧠 K-Means Clustering for Product Segmentation**

**🏷️ Category-Based Performance Analysis**

**💬 Review Sentiment Word Cloud**

**✅ Actionable Business Recommendations**

It was designed to help businesses:

- Optimize product pricing strategies

- Identify top-performing categories

- Understand customer sentiment

- Maximize ROI through data-driven insights

---

## 📬 Author

**Gbenga Kajola**
🎓 Certified Data Analyst | 👨‍💻 Certified Data Scientist | 🧠 AI/ML Engineer | 📱 Mobile App Developer 

[LinkedIn](https://www.linkedin.com/in/kajolagbenga)

[Portfolio](https://kajolagbenga.netlify.app)

[Certified_Data_Scientist](https://www.datacamp.com/certificate/DSA0012312825030)

[Certified_Data_Analyst](https://www.datacamp.com/certificate/DAA0018583322187)

[Certified_SQL_Database_Programmer](https://www.datacamp.com/certificate/SQA0019722049554)

---

## 🔧 Features

- **Interactive Filtering:** Category, price range, and rating range selection.
- **KPIs:** Average rating, total reviews, and average discount.
- **Prescriptive Insights:** Recommendations based on product ratings, discounts, and review volume.
- **Visual Analytics:** 
  - Discount distribution histogram
  - Rating vs Discount/Price scatter plots
  - Category performance bar charts
- **Clustering Analysis:** K-Means segmentation based on price, rating, and review count.
- **Sentiment Mining:**
  - Word cloud of customer reviews
  - Top positive and negative keywords from review content

---


## 🧾 Dataset

The dashboard uses an `amazon.csv` file with the following key columns:

- `product_name`, `category`, `discounted_price`, `actual_price`, `discount_percentage`
- `rating`, `rating_count`, `review_content`

Ensure this file is placed in the same directory as the app script.

---


## 🚀 Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/amazon-prescriptive-dashboard.git
cd amazon-prescriptive-dashboard
```

### 2. Install required libraries

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run your_dashboard_script.py
```

> Replace `your_dashboard_script.py` with your actual script file name if different.

---

## 📋 requirements.txt

```
streamlit
pandas
numpy
matplotlib
seaborn
wordcloud
scikit-learn
plotly
```

---

## 💡 Use Cases

- E-commerce product managers seeking pricing optimization insights.
- Marketers looking to understand review sentiment and customer preferences.
- Data analysts performing category-level product performance evaluation.

---

## 🔐 License

This project is licensed under the MIT License.