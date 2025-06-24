# 📦 Amazon Product Prescriptive Analytics Dashboard

This interactive Streamlit dashboard provides comprehensive **descriptive, diagnostic, and prescriptive analytics** for Amazon product data, allowing users to filter by product category, price, and rating, while exploring visual insights and actionable recommendations.

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

## 🧾 Dataset

The dashboard uses an `amazon.csv` file with the following key columns:

- `product_name`, `category`, `discounted_price`, `actual_price`, `discount_percentage`
- `rating`, `rating_count`, `review_content`

Ensure this file is placed in the same directory as the app script.

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

## 📊 Screenshots

| Metric Cards | Discount Analysis | Clustering |
|--------------|-------------------|------------|
| ![metrics](screenshots/metrics.png) | ![discount](screenshots/discount.png) | ![cluster](screenshots/cluster.png) |

> _You can create a `screenshots/` folder and save screenshots of your dashboard._

## 💡 Use Cases

- E-commerce product managers seeking pricing optimization insights.
- Marketers looking to understand review sentiment and customer preferences.
- Data analysts performing category-level product performance evaluation.

## 🔐 License

This project is licensed under the MIT License.