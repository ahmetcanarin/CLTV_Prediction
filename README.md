# CLTV Prediction with BG-NBD and Gamma-Gamma

**Business Problem**

FLO wants to determine a roadmap for its sales and marketing activities.
In order for the company to make medium- and long-term plans, it is necessary to estimate the potential future value that existing customers will generate for the company.

**Dataset Story**

The dataset consists of information obtained from the past shopping behaviors of customers who made their most recent purchases in 2020–2021 via OmniChannel (both online and offline shopping).

**master_id**: Unique customer ID
**order_channel**: The channel/platform used for shopping (Android, iOS, Desktop, Mobile, Offline)
**last_order_channel**: The channel where the most recent purchase was made
**first_order_date**: The date of the customer’s first purchase
**last_order_date**: The date of the customer’s most recent purchase
**last_order_date_online**: The date of the customer’s most recent online purchase
**last_order_date_offline**: The date of the customer’s most recent offline purchase
**order_num_total_ever_online**: Total number of purchases made by the customer on the online platform
**order_num_total_ever_offline**: Total number of purchases made by the customer offline
**customer_value_total_ever_offline**: Total amount paid by the customer for offline purchases
**customer_value_total_ever_online**: Total amount paid by the customer for online purchases
**interested_in_categories_12**: List of categories the customer has shopped in over the last 12 months
