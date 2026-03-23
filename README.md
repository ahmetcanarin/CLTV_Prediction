# 📊 CLTV Prediction with BG-NBD and Gamma-Gamma

---

## 🧩 Business Problem

FLO aims to create a strategic roadmap for its sales and marketing activities.

To support **medium- and long-term planning**, it is essential to estimate the **potential future value (Customer Lifetime Value)** that existing customers will generate.

This enables:
- More accurate budgeting and forecasting  
- Data-driven marketing strategies  
- Improved customer relationship management  

---

## 📁 Dataset Story

The dataset consists of customer shopping behavior data from **2020–2021**, covering **OmniChannel** activity (both online and offline).

---

## 📌 Variables

| Variable | Description |
|----------|-------------|
| **master_id** | Unique customer ID |
| **order_channel** | The channel/platform used for shopping (Android, iOS, Desktop, Mobile, Offline) |
| **last_order_channel** | The channel of the most recent purchase |
| **first_order_date** | The date of the first purchase |
| **last_order_date** | The date of the most recent purchase |
| **last_order_date_online** | The date of the most recent online purchase |
| **last_order_date_offline** | The date of the most recent offline purchase |
| **order_num_total_ever_online** | Total number of online purchases |
| **order_num_total_ever_offline** | Total number of offline purchases |
| **customer_value_total_ever_offline** | Total amount spent on offline purchases |
| **customer_value_total_ever_online** | Total amount spent on online purchases |
| **interested_in_categories_12** | Categories shopped in during the last 12 months |

---

## 🎯 Project Objectives

- Estimate **Customer Lifetime Value (CLTV)** using probabilistic models  
- Predict future purchasing behavior  
- Identify high-value customers  
- Support targeted marketing strategies  

---

## 🛠️ Methodology

- Data preprocessing and feature engineering  
- Creation of CLTV dataset structure  
- Model implementation:
  - **BG/NBD (Beta-Geometric / Negative Binomial Distribution)** → purchase frequency prediction  
  - **Gamma-Gamma Model** → monetary value prediction  
- CLTV calculation (6-month projection)  
- Customer segmentation based on CLTV  

---

## 🚀 Expected Outcome

- Accurate estimation of customer lifetime value  
- Identification of high-potential customer segments  
- Improved marketing efficiency and ROI  
- Better long-term strategic planning  
