import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
def load_data():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    sql_query = "SELECT * FROM public.products_catalog"
    df = pd.read_sql(sql_query, engine)
    return (df)

# Main function
def main():
    st.title('Interactive Streamlit Dashboard')

    # Load data
    global products_catalog
    if 'products_catalog' not in st.session_state:
        st.session_state.products_catalog = load_data()
        # remove outliers for viz purposes
        q_high = st.session_state.products_catalog["price"].quantile(0.95)
        q_low = st.session_state.products_catalog["price"].quantile(0.05)
        st.session_state.products_catalog = st.session_state.products_catalog[(st.session_state.products_catalog["price"] < q_high) & 
                                                                              (st.session_state.products_catalog["price"] > q_low)]

    fig = px.box(st.session_state.products_catalog, 
                       x="price", 
                       color="catalog_id")
    st.plotly_chart(fig)    

    fig = px.pie(st.session_state.products_catalog, 
                 values="price", 
                 names="catalog_id",
                 title= "Sum (prices) per catalog_id")
    st.plotly_chart(fig) 

    fig = px.pie(st.session_state.products_catalog.groupby(["catalog_id"])["product_id"].count().reset_index(), #.rename_axis("limbs", axis="columns"), 
                 values="product_id", 
                 names="catalog_id",
                 title= "Count (products) per catalog_id")
    st.plotly_chart(fig) 

    fig = px.pie(st.session_state.products_catalog.groupby(["size_title"])["price"].sum().sort_values(ascending = False).head(10).reset_index(), 
                 values="price", 
                 names="size_title",
                 title= "Sum (price) per size_title")
    st.plotly_chart(fig) 

    fig = px.pie(st.session_state.products_catalog.groupby(["size_title"])["product_id"].count().sort_values(ascending = False).head(10).reset_index(),
                 values="product_id", 
                 names="size_title",
                 title= "Count (products) per size_title")
    st.plotly_chart(fig) 

    st.table(st.session_state.products_catalog.groupby(["size_title"])["price"].mean().sort_values(ascending = False).head(10).reset_index())

    fig = px.pie(st.session_state.products_catalog.groupby(["brand_title"])["price"].sum().sort_values(ascending = False).head(10).reset_index(), 
                 values="price", 
                 names="brand_title",
                 title= "Sum (price) per brand_title")
    st.plotly_chart(fig) 

    fig = px.pie(st.session_state.products_catalog.groupby(["brand_title"])["product_id"].count().sort_values(ascending = False).head(10).reset_index(),
                 values="product_id", 
                 names="brand_title",
                 title= "Count (products) per brand_title")
    st.plotly_chart(fig) 

    st.table(st.session_state.products_catalog.groupby(["brand_title"])["price"].mean().sort_values(ascending = False).head(10).reset_index())

    



if __name__ == "__main__":
    main()