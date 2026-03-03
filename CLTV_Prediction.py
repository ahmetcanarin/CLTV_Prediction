##################################
# Data Preparation
##################################

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("CRM Analytics/Ödev/flo_data_20k.csv")
df = df_.copy()

df.describe().T
df.isnull().sum()
df.isnull().values.any()
df.nunique()
df.shape


# Defining the outlier_thresholds and replace_with_thresholds functions required to trim outliers.
# Note: While calculating CLTV, frequency values must be integers. Therefore, it should be rounded the lower and upper limits using round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Trimming the outliers of the variables
# "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online".

df.describe().T

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)


# Omnichannel customers are those who shop both online and offline.
# Creating new variables for each customer's total number of purchases and total spending.

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# Examining the variable types and convert the date-related variables to datetime format.

df.info()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


###############################################################
# CLTV Data Structure Creation
###############################################################

# Taking two days after the most recent purchase date in the dataset as the analysis date.

df["last_order_date"].max()     # 2021-05-30

today_date = df["last_order_date"].max() + dt.timedelta(days=2)


# Creating a new CLTV dataframe that includes customer_id, recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
cltv_df["T_weekly"] = (today_date - df["first_order_date"]).dt.days / 7
cltv_df["frequency"] = df["total_order_num"]
cltv_df["monetary_cltv_avg"] = df["total_customer_value"] / df["total_order_num"]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]


###########################################################################
# Building the BG/NBD and Gamma-Gamma Models and Calculating 6-Month CLTV
###########################################################################

# Building the BG/NBD model.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])


# Predicting the expected purchases from customers within 3 months and add them to the CLTV dataframe as exp_sales_3_month.

# bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
#                                                         cltv_df["frequency"],
#                                                         cltv_df["recency_cltv_weekly"],
#                                                         cltv_df["T_weekly"]).sort_values(ascending=False)

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"]).sort_values(ascending=False)


# Predicting the expected purchases from customers within 6 months and add them to the CLTV dataframe as exp_sales_6_month.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"]).sort_values(ascending=False)

plot_period_transactions(bgf)
plt.show()


# Examining the top 10 customers who are expected to make the most purchases in the 3rd and 6th months.

cltv_df.sort_values(["exp_sales_3_month", "exp_sales_6_month"], ascending=False).head(10)


# Fitting the Gamma-Gamma model. Estimating the average expected profit per customer and add it to the CLTV dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])


# Calculating the 6-month CLTV and add it to the dataframe as cltv.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv


# Observing the top 20 customers with the highest CLTV values.

cltv_df.sort_values("cltv", ascending=False).head(20)


###############################################################
# Creating Segments Based on CLTV
###############################################################

# Based on the 6-month CLTV, we divide all customers into 4 groups (segments) and add the segment labels to the dataset.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df["cltv_segment"].value_counts()

# Examining the average recency, frequency, and monetary values of the segments.

cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly": "mean",
                                    "frequency": "mean",
                                    "monetary_cltv_avg": "mean"})




