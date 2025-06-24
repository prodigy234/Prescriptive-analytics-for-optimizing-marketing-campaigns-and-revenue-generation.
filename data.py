# STREAMLIT ADVANCED PRESCRIPTIVE ANALYTICS DASHBOARD

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from io import BytesIO
import plotly.express as px

# Load dataset
data = pd.read_csv('amazon.csv')

# Data Cleaning
for col in ['discounted_price', 'actual_price']:
    data[col] = data[col].str.replace('\u20b9', '').str.replace(',', '').astype(float)
data['discount_percentage'] = data['discount_percentage'].str.replace('%', '').astype(float)
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data['rating_count'] = data['rating_count'].str.replace(',', '').astype(float)
data.dropna(subset=['rating', 'rating_count'], inplace=True)

# --- STREAMLIT UI SETUP --- #
st.set_page_config(page_title="Amazon Product Insights", layout="wide", page_icon="📦")
st.title("📦 Advanced Prescriptive Analytics Dashboard")
st.markdown("Explore **product performance, pricing strategy, sentiment**, and **category ROI recommendations** using Amazon product sales data.")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
categories = st.sidebar.multiselect("Select Categories", options=data['category'].unique(), default=data['category'].unique())
price_range = st.sidebar.slider("Discounted Price Range", float(data['discounted_price'].min()), float(data['discounted_price'].max()), (float(data['discounted_price'].min()), float(data['discounted_price'].max())))

filtered_data = data[(data['category'].isin(categories)) &
                     (data['discounted_price'] >= price_range[0]) &
                     (data['discounted_price'] <= price_range[1])]

# --- METRICS --- #
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg. Rating", round(filtered_data['rating'].mean(), 2))
col2.metric("💬 Total Reviews", int(filtered_data['rating_count'].sum()))
col3.metric("🏷️ Avg. Discount %", round(filtered_data['discount_percentage'].mean(), 2))

# --- DISCOUNT DISTRIBUTION --- #
st.subheader("🎯 Discount Strategy")
fig1 = px.histogram(filtered_data, x='discount_percentage', nbins=20, title="Distribution of Discount Percentages")
st.plotly_chart(fig1, use_container_width=True)

# --- SCATTER PLOTS --- #
st.subheader("📊 Impact of Pricing & Discounts on Ratings")
col4, col5 = st.columns(2)
with col4:
    fig2 = px.scatter(filtered_data, x='discount_percentage', y='rating', size='rating_count', color='rating_count', title="Discount % vs. Rating")
    st.plotly_chart(fig2, use_container_width=True)
with col5:
    fig3 = px.scatter(filtered_data, x='discounted_price', y='rating', size='rating_count', color='rating_count', title="Price vs. Rating")
    st.plotly_chart(fig3, use_container_width=True)

# --- CATEGORY ANALYSIS --- #
category_summary = filtered_data.groupby('category').agg({'rating': 'mean', 'discount_percentage': 'mean', 'rating_count': 'sum'}).reset_index()

st.subheader("🏆 Category Performance")
fig4 = px.bar(category_summary.sort_values(by='rating', ascending=False), x='rating', y='category', color='rating', orientation='h', title="Average Ratings by Category")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.bar(category_summary.sort_values(by='rating_count', ascending=False), x='rating_count', y='category', color='discount_percentage', orientation='h', title="Rating Counts vs. Discount % by Category")
st.plotly_chart(fig5, use_container_width=True)

# --- PRESCRIPTIVE INSIGHTS --- #
st.subheader("🧠 AI-Powered Prescriptive Recommendations")
high_rating_categories = category_summary[category_summary['rating'] >= 4.0].sort_values('rating_count', ascending=False)
st.markdown("### ✅ Best Performing Categories")
st.dataframe(high_rating_categories)
st.markdown("#### Recommendations")
st.markdown("""
- **Target discounts between 30%-60%** to boost perceived value.
- **Promote categories** with high average ratings (≥ 4.0) and strong review counts for higher ROI.
- **Bundle products** in top-rated categories to increase upsell chances.
- **Monitor complaints in reviews** to improve product quality and reduce negative reviews.
""")

# --- WORD CLOUD --- #
st.subheader("🔍 Customer Review Word Cloud")
review_text = " ".join(filtered_data['review_content'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(review_text)
fig6, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig6)

# --- SENTIMENT INSIGHT --- #
st.subheader("📣 Sentiment Signals")
cv = CountVectorizer(stop_words='english', max_features=1000)
words = cv.fit_transform(filtered_data['review_content'].dropna())
word_freq = dict(zip(cv.get_feature_names_out(), words.toarray().sum(axis=0)))
top_positive = Counter({k: v for k, v in word_freq.items() if 'good' in k or 'great' in k}).most_common(10)
top_negative = Counter({k: v for k, v in word_freq.items() if 'bad' in k or 'poor' in k}).most_common(10)

col6, col7 = st.columns(2)
with col6:
    st.markdown("### 👍 Top Positive Words")
    st.table(top_positive)
with col7:
    st.markdown("### 👎 Top Negative Words")
    st.table(top_negative)

# --- END OF DASHBOARD --- #
