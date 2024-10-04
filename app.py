import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import datetime
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


st.markdown("""
    <style>
    /* Sidebar background with gradient and rounded corners */
    .css-1d391kg {
        background: linear-gradient(135deg, #00796B, #004D40) !important; /* Darker teal gradient */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); /* Enhanced shadow */
        padding: 20px; /* More padding for spacious look */
    }

    /* Sidebar text color */
    .css-10trblm {
        color: #FFFFFF !important; /* White text for better contrast */
        font-weight: bold; /* Bold text for emphasis */
        font-size: 1.1em; /* Slightly larger font size */
    }

    /* Input backgrounds */
    .stTextInput, .stTextArea {
        background-color: #E0F2F1 !important; /* Light input background */
        border-radius: 5px; /* Rounded input corners */
        padding: 10px; /* Padding for input elements */
    }

    /* Radio button style */
    input[type="radio"] {
        accent-color: #00796B; /* Darker teal for radio buttons */
    }

    /* Button style */
    .stButton > button {
        background-color: #00796B; /* Teal buttons */
        color: #FFFFFF; /* White text */
        border-radius: 5px; /* Rounded button corners */
        padding: 10px 15px; /* Added padding for buttons */
    }

    /* Main content background and text */
    .css-12oz5g7 {
        background-color: #F1F8E9 !important; /* Light greenish background for content */
    }

    /* Heading style */
    h1, h2, h3 {
        color: #00796B; /* Darker teal heading colors */
    }

    /* Horizontal line color */
    hr {
        border: 1px solid #B2DFDB; /* Light horizontal lines */
    }
    </style>
    """, unsafe_allow_html=True)
file_path = 'star_schema_rfm_categories_updated.xlsx'
rfm_data = pd.read_excel(file_path, sheet_name='Fact with RFM Categories')
# Load the CSV data


# # Set the Streamlit page configuration
# st.set_page_config(page_title="Customer Insights Dashboard", initial_sidebar_state='expanded')

# Load Excel file and get all sheet names
excel_file = 'RFM Analysis.xlsx'
xls = pd.ExcelFile(excel_file)

# Load each sheet into a DataFrame
Transactions_df = pd.read_excel(xls, 'Transactions')
NewCustomerList_df = pd.read_excel(xls, 'NewCustomerList')
CustomerDemographic_df = pd.read_excel(xls, 'CustomerDemographic')
CustomerAddress_df = pd.read_excel(xls, 'CustomerAddress')

# Set Streamlit page configuration
#st.set_page_config(page_title="Customer Data Dashboard", layout='wide')

# Add a sidebar for navigation
section = st.sidebar.radio("Navigate", ["Home", "Data", "Prediction & Classification"])
# Welcome Page
from PIL import Image

if section == "Home":

    # Home Page - Introduction
    st.title("Welcome to the RFM Project")

    # Load and display the image
    img = Image.open("mohamedhesham.jpg")
    st.image(img, caption="Mohamed Hesham - Passionate Data Scientist")
    img2 = Image.open("mohamedhesham2.jpg")
    st.image(img2, caption="Mohamed Hesham - Passionate Data Scientist")

    # Introduction about yourself
    st.write("""
    ### About Me
    Hi, I'm **Mohamed Hesham**, a passionate data scientist with a deep interest in leveraging data to drive insights and decisions. 
    With a strong foundation in data analysis, I am constantly exploring new ways to enhance business operations through data-driven strategies.
    """)

    # RFM Data Transformation Process
    st.write("""
    ### RFM Data Transformation Process to Star Schema
    This process outlines how raw transactional and customer data from the RFM Analysis is transformed into a star schema format. 
    The star schema structure enables efficient data querying and analysis, providing insights into customer behavior through Recency, Frequency, and Monetary (RFM) analysis.
    """)

    # Process explanation
    st.write("""
    **Step 1: Data Ingestion**  
    The process starts with the ingestion of raw data from multiple sources:  
    - **Transactions Data**: Contains product purchases, customer details, and financial metrics (e.g., list price, order status).  
    - **Customer Demographics**: Includes demographic data like name, gender, and job title.  
    - **Customer Addresses**: Provides customer address information.  
    These raw datasets will be used to calculate RFM scores for customer segmentation.

    **Step 2: RFM Calculation**  
    RFM analysis involves calculating the following metrics for each customer:  
    - **Recency**: The time since the customer's last purchase.  
    - **Frequency**: The number of purchases made by the customer.  
    - **Monetary**: The total value of the purchases.  
    These metrics are then combined into a single RFM Score for each customer, which is further categorized into segments.

    **Step 3: Customer Segmentation**  
    Once the RFM scores are calculated, customers are segmented based on their scores:  
    - **RFM Score**: A combined score used to categorize customers into different groups.  
    - **Segments**: These segments represent different customer behaviors, such as 'Lost Customers', 'Frequent Buyers', etc.  
    This segmentation allows businesses to tailor their marketing strategies based on customer behavior.

    **Step 4: Building the Star Schema**  
    The final step involves transforming the raw data and RFM analysis into a star schema structure. The star schema includes:  
    - **Customer Dimension**: Contains customer demographic details.  
    - **Product Dimension**: Includes product-related information.  
    - **Date Dimension**: Represents transaction dates for time-based analysis.  
    - **Address Dimension**: Stores customer addresses.  
    - **Fact Table (with RFM Categories)**: Stores detailed transaction data along with RFM scores and categories.
    """)
    # Display RFM Pie Chart
    img_pie = Image.open("png1.png")
    st.image(img_pie, caption="RFM Customer Segmentation")
    # Conclusion and Columns Overview
    st.write("""
    ### Conclusion
    This process allows businesses to gain valuable insights into customer behavior by leveraging RFM analysis and organizing the data in a star schema format. 
    This enables efficient querying and analysis, facilitating more informed decision-making and targeted marketing strategies.

    ### Data Columns Overview
    Here are the key columns from the datasets that are used in the RFM analysis process:
    - **Transactions Data**:
      - transaction_id, product_id, customer_id, transaction_date, online_order, list_price, product_first_sold_date  
    - **Customer Demographics**:
      - customer_id, first_name, last_name, gender, DOB, past_3_years_bike_related_purchases, job_title  
    - **Customer Addresses**:
      - customer_id, address, postcode, state, country  

    ### RFM Calculation Equations
    To calculate the RFM metrics for each customer, we use the following formulas:
    - **Recency**: Days since the customer's last purchase (e.g., Today’s Date - Last Purchase Date).
    - **Frequency**: Number of purchases made by the customer.
    - **Monetary**: Total value of purchases made by the customer (sum of transaction amounts).
    These metrics are combined into an RFM score and used for customer segmentation.
    """)

    # Load and display the star schema image
    st.write("### RFM Star Schema")
    star_schema_img = Image.open("11.jpg")
    st.image(star_schema_img, caption="RFM Star Schema Overview")

    # Load and display Power BI images
    st.write("### Power BI Dashboard")
    power_bi_img1 = Image.open("22.jpg")
    st.image(power_bi_img1, caption="Power BI Dashboard View 1")

    power_bi_img2 = Image.open("33.jpg")
    st.image(power_bi_img2, caption="Power BI Dashboard View 2")

    power_bi_img3 = Image.open("44.jpg")
    st.image(power_bi_img3, caption="Power BI Dashboard View 3")
    st.write("""
    The Power BI dashboards visualize the insights gained from the RFM analysis, allowing stakeholders to interactively explore customer segments and behaviors.
    These visualizations can drive strategic decision-making and enhance understanding of customer dynamics.
    """)








# Tables section
elif section == "Data":
    table = st.sidebar.radio("Choose a table", ["Transactions", "New Customer List", "Customer Demographic", "Customer Address"])
    # Transactions page
    if table == "Transactions":
        st.header("Transactions Data")

        st.write("### **Transactions Metadata**")
        transaction_meta = pd.DataFrame({
            'Column': ['transaction_id', 'product_id', 'customer_id', 'transaction_date', 'online_order', 'order_status',
                       'brand', 'product_line', 'product_class', 'product_size', 'list_price', 'standard_cost',
                       'product_first_sold_date'],
            'Datatype': ['Number', 'Number', 'Number', 'Date', 'Boolean', 'Text', 'Text', 'Text', 'Text', 'Text',
                         'Currency', 'Currency', 'Date'],
            'Description': [
                'Unique identifier for each transaction (Primary Key)',
                'Product identifier (Foreign Key)',
                'Customer identifier (Foreign Key)',
                'Date of transaction',
                'True/False if the order was online',
                'Order status (Approved/Cancelled)',
                'Product brand',
                'Product line (e.g., Road, Touring)',
                'Product classification (e.g., high, medium, low)',
                'Product size (small, medium, large)',
                'Product list price',
                'Standard cost of the product',
                'Date when the product was first sold'
            ]
        })
        st.dataframe(transaction_meta, use_container_width=True)

        # Display sample data
        st.write("### **Sample Transaction Data**")
        st.dataframe(Transactions_df.sample(10), use_container_width=True)
        st.write(f"#### Dataframe Shape: **{Transactions_df.shape}**")

    # New Customer List page
    elif table == "New Customer List":
        st.header("New Customer List Data")

        st.write("### **New Customer List Metadata**")
        new_customer_meta = pd.DataFrame({
            'Column': ['first_name', 'last_name', 'gender', 'past_3_years_bike_related_purchases', 'DOB', 'job_title',
                       'job_industry_category', 'wealth_segment', 'deceased_indicator', 'owns_car', 'tenure', 'address',
                       'postcode', 'state', 'country', 'property_valuation', 'Rank', 'Value'],
            'Datatype': ['Text', 'Text', 'Text', 'Number', 'Date', 'Text', 'Text', 'Text', 'Text', 'Boolean', 'Number',
                         'Text', 'Text', 'Text', 'Text', 'Number', 'Number', 'Currency'],
            'Description': [
                'Customer\'s first name',
                'Customer\'s last name',
                'Customer\'s gender',
                'Number of bike-related purchases in the last 3 years',
                'Customer\'s date of birth',
                'Customer\'s job title',
                'The industry category in which the customer works',
                'Classification based on customer\'s wealth (Mass, Affluent, High Net Worth)',
                'Indicates if the customer is deceased (Y/N)',
                'Indicates if the customer owns a car (Yes/No)',
                'The length of time (in years) the customer has been associated with the store.',
                'Customer\'s full address',
                'Postal code of the customer\'s address',
                'State of residence',
                'Country of residence (Australia)',
                'Numeric property valuation rating (1-12)',
                'Customer ranking score',
                'Total customer value'
            ]
        })
        st.dataframe(new_customer_meta, use_container_width=True)

        # Display sample data
        st.write("### **Sample New Customer Data**")
        st.dataframe(NewCustomerList_df.sample(10), use_container_width=True)
        st.write(f"#### Dataframe Shape: **{NewCustomerList_df.shape}**")

    # Customer Demographic page
    elif table == "Customer Demographic":
        st.header("Customer Demographic Data")

        st.write("### **Customer Demographic Metadata**")
        demographic_meta = pd.DataFrame({
            'Column': ['customer_id', 'first_name', 'last_name', 'gender', 'past_3_years_bike_related_purchases', 'DOB',
                       'job_title', 'job_industry_category', 'wealth_segment', 'deceased_indicator', 'owns_car', 'tenure'],
            'Datatype': ['Number', 'Text', 'Text', 'Text', 'Number', 'Date', 'Text', 'Text', 'Text', 'Boolean', 'Boolean',
                         'Number'],
            'Description': [
                'Unique identifier for customers (Primary Key)',
                'Customer\'s first name',
                'Customer\'s last name',
                'Customer\'s gender',
                'Number of bike-related purchases in the last 3 years',
                'Customer\'s date of birth',
                'Customer\'s job title',
                'The industry category in which the customer works',
                'Classification based on customer\'s wealth (Mass, Affluent, High Net Worth)',
                'Indicates if the customer is deceased (Y/N)',
                'Indicates if the customer owns a car (Yes/No)',
                'The length of time (in years) the customer has been associated with the store.'
            ]
        })
        st.dataframe(demographic_meta, use_container_width=True)

        # Display sample data
        st.write("### **Sample Customer Demographic Data**")
        st.dataframe(CustomerDemographic_df.sample(10), use_container_width=True)
        st.write(f"#### Dataframe Shape: **{CustomerDemographic_df.shape}**")

    # Customer Address page
    elif table== "Customer Address":
        st.header("Customer Address Data")

        st.write("### **Customer Address Metadata**")
        address_meta = pd.DataFrame({
            'Column': ['customer_id', 'address', 'postcode', 'state', 'country', 'property_valuation'],
            'Datatype': ['Number', 'Text', 'Text', 'Text', 'Text', 'Number'],
            'Description': [
                'Unique identifier for customers (Foreign Key)',
                'Customer\'s full address',
                'Postal code of the customer\'s address',
                'State of residence',
                'Country of residence (Australia)',
                'Numeric property valuation rating (1-12)'
            ]
        })
        st.dataframe(address_meta, use_container_width=True)

        # Display sample data
        st.write("### **Sample Customer Address Data**")
        st.dataframe(CustomerAddress_df.sample(10), use_container_width=True)
        st.write(f"#### Dataframe Shape: **{CustomerAddress_df.shape}**")


# Prediction & Classification section
elif section == "Prediction & Classification":
    model = st.sidebar.radio("Choose a model", ["Product Recommendation", "Customer Clustering", "Churn Prediction"])

    # Product Recommendation Model
    if model == "Product Recommendation":
        st.title("Product Recommendation System")

        from sklearn.decomposition import TruncatedSVD
        from scipy.sparse import csr_matrix
        import numpy as np


        # Assuming `rfm_data` is already loaded with the necessary customer-product interaction data.
        interaction_matrix = rfm_data.pivot_table(index='customer_id', columns='product_id', values='monetary',
                                                  fill_value=0)
        sparse_matrix = csr_matrix(interaction_matrix)

        svd = TruncatedSVD(n_components=50, random_state=42)
        decomposed_matrix = svd.fit_transform(sparse_matrix)
        correlation_matrix = np.corrcoef(decomposed_matrix)


        def recommend_products_fixed(customer_id, interaction_matrix, correlation_matrix):
            customer_index = interaction_matrix.index.get_loc(customer_id)
            customer_purchases = interaction_matrix.iloc[customer_index]
            purchased_products = customer_purchases[customer_purchases > 0].index.tolist()

            recommendations = []
            for product_id in purchased_products:
                if product_id in interaction_matrix.columns:
                    product_index = interaction_matrix.columns.get_loc(product_id)
                    if product_index < len(correlation_matrix):
                        similar_products = list(enumerate(correlation_matrix[product_index]))
                        similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)

                        for product, score in similar_products:
                            if product < len(interaction_matrix.columns) and interaction_matrix.columns[
                                product] not in purchased_products and score > 0:
                                recommendations.append(interaction_matrix.columns[product])

            return list(set(recommendations))[:5]


        customer_id = st.number_input("Enter Customer ID:", min_value=1)
        recommended_products = recommend_products_fixed(customer_id, interaction_matrix, correlation_matrix)

        # Display the recommended products
        st.subheader("🌟 Your Top Product Recommendations 🌟")

        if recommended_products:
            st.write("Here are some products we think you'll love:")

            # Use bullet points to list recommended products neatly
            for product_id in recommended_products:
                st.write(f"- **Product ID:** {str(product_id)}")  # Converting int64 to str to avoid int64 type display
        else:
            st.write("No recommendations available for this customer. Please try a different Customer ID.")

    # Customer Clustering Model
    elif model == "Customer Clustering":
        st.title("Customer Clustering")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import matplotlib.pyplot as plt
        import pandas as pd

        # Step 1: Prepare the data for clustering
        rfm_features = rfm_data[['recency', 'frequency', 'monetary']]
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features)

        # Step 2: Dimensionality Reduction using PCA
        pca = PCA(n_components=2)
        rfm_pca = pca.fit_transform(rfm_scaled)

        # Step 3: Apply KMeans Clustering
        kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
        rfm_data['Cluster'] = kmeans_model.fit_predict(rfm_pca)

        # Step 4: Evaluate the Model
        silhouette_avg = silhouette_score(rfm_pca, rfm_data['Cluster'])
        st.write(f"**Silhouette Score for K-Means:** {silhouette_avg:.2f}")

        # Step 5: Cluster Analysis
        st.subheader("📊 Cluster Analysis:")
        st.write(
            "In this section, we present an analysis of the identified customer clusters based on their recency, frequency, and monetary values. Each cluster represents a group of customers with similar purchasing behaviors.")

        cluster_summaries = []
        cluster_attributes = {
            0: "New Customers: Typically characterized by high recency, indicating they have recently made their first purchases. These customers may need encouragement to become repeat buyers.",
            1: "Loyal Customers: Known for their frequent purchases and high monetary values. These customers are engaged with your brand and are likely to respond well to loyalty programs.",
            2: "At-Risk Customers: Display low frequency and high recency, suggesting they haven't purchased in a while. These customers may require targeted re-engagement strategies."
        }

        for cluster in rfm_data['Cluster'].unique():
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
            cluster_summary = {
                'Cluster': cluster,
                'Number of Customers': len(cluster_data),
                'Average Recency': round(cluster_data['recency'].mean(), 2),
                'Average Frequency': round(cluster_data['frequency'].mean(), 2),
                'Average Monetary': round(cluster_data['monetary'].mean(), 2),
                'Attributes': cluster_attributes[cluster]  # Add cluster attributes
            }
            cluster_summaries.append(cluster_summary)

        # Display cluster summaries in a DataFrame
        cluster_summary_df = pd.DataFrame(cluster_summaries)
        st.table(cluster_summary_df)

        st.write(
            "The table above summarizes the characteristics of each cluster. Understanding these attributes can help you tailor your marketing strategies effectively.")

        # Step 6: Visualize Clusters
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm_data['Cluster'], cmap='viridis', alpha=0.6)
        plt.title('K-Means Clusters', fontsize=16)
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.colorbar(scatter, label='Cluster ID')  # Add color bar for clusters
        st.pyplot(plt)

        st.write(
            "The scatter plot visualizes the customer clusters in a two-dimensional space, allowing you to see the distribution of customers based on their RFM scores. Each color represents a different cluster, illustrating how customers are grouped based on their purchasing behavior.")


    # Churn Prediction Model
    elif model == "Churn Prediction":
        st.title("Churn Prediction Model")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd

        # Step 1: Create a Churn Target Variable
        rfm_data['churn'] = (rfm_data['recency'] > 100).astype(int)

        # Features and target variable for churn prediction
        X_churn = rfm_data[['recency', 'frequency', 'monetary']]
        y_churn = rfm_data['churn']

        # Step 2: Split the Data into Training and Testing Sets
        X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
            X_churn, y_churn, test_size=0.2, random_state=42
        )

        # Step 3: Train a Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_churn, y_train_churn)

        # Step 4: Make Predictions
        y_pred_churn = rf_classifier.predict(X_test_churn)

        # Step 5: Evaluate the Model
        accuracy_churn = accuracy_score(y_test_churn, y_pred_churn)
        classification_report_churn = classification_report(y_test_churn, y_pred_churn,
                                                            target_names=['Not Churned', 'Churned'])

        # Displaying the Results
        st.subheader("🔍 Model Evaluation:")
        st.write(
            "The churn prediction model helps identify customers who are likely to stop purchasing your products. Understanding churn is vital for businesses as it helps in developing strategies to retain valuable customers.")

        st.write(f"**Accuracy of Churn Prediction Model:** {accuracy_churn:.2%}")
        st.write("### Classification Report:")
        st.text(classification_report_churn)

        st.write("### Key Insights from the Classification Report:")
        st.write(
            "- **Precision**: The ratio of true positive predictions to the total predicted positives. A higher precision indicates fewer false positives."
        )
        st.write(
            "- **Recall**: The ratio of true positive predictions to the actual positives. A higher recall indicates fewer false negatives."
        )
        st.write(
            "- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics."
        )
        st.write(
            "- **Support**: The number of actual occurrences in each class. This helps in understanding the distribution of classes."
        )

        st.write(
            "By analyzing these metrics, you can gain insights into how well your model is performing and where improvements might be needed.")

st.sidebar.write("---")
st.sidebar.markdown("Crafted with passion and precision by Mohamed Hesham 🚀")
