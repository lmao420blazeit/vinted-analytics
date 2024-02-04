import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import subprocess

# Load a sample dataset
def load_data():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    sql_query = "SELECT * FROM public.products_catalog LIMIT 5000"
    df = pd.read_sql(sql_query, engine)
    return (df)

# Main function
def main():
    st.set_page_config(layout="wide")

    # Load data
    global products_catalog
    if 'products_catalog' not in st.session_state:
        st.session_state.products_catalog = load_data()
        # remove outliers for viz purposes
        q_high = st.session_state.products_catalog["price"].quantile(0.95)
        q_low = st.session_state.products_catalog["price"].quantile(0.05)
        st.session_state.products_catalog = st.session_state.products_catalog[(st.session_state.products_catalog["price"] < q_high) & 
                                                                              (st.session_state.products_catalog["price"] > q_low)]
        
    st.title("Brands")

    price = st.session_state.products_catalog.groupby(["brand_title"])["price"].sum()
    price = price/price.sum()

    products = st.session_state.products_catalog.groupby(["brand_title"])["product_id"].count()
    products = products/products.sum()
    fig = go.Figure(data=[
        go.Bar(name='Price', 
               x=price.sort_values(ascending = False).head(20).reset_index()["brand_title"], 
               y=price.sort_values(ascending = False).head(20).reset_index()["price"]
               ),
        go.Bar(name='Count', 
               x=products.sort_values(ascending = False).head(20).reset_index()["brand_title"], 
               y=products.sort_values(ascending = False).head(20).reset_index()["product_id"])
    ])
    fig.update_layout(title = "Normalized price and count")
    st.plotly_chart(fig, use_container_width=True) 

    col1, col2 = st.columns([0.7, 0.3])

    # do median with boxplot
    # do stacked bar with percentage of total count and price

    brands = st.session_state.products_catalog.groupby(["brand_title"])["product_id"].count().sort_values(ascending = False).head(20).reset_index()["brand_title"]

    fig = px.box(st.session_state.products_catalog[st.session_state.products_catalog["brand_title"].isin(brands)], 
                       y="price", 
                       color="brand_title",
                       orientation="v",
                       title= "Brand boxplot")
    
    with col1:
        st.plotly_chart(fig, use_container_width=True) 

    fig = px.pie(price, 
                 values=price.sort_values(ascending = False).head(20).reset_index()["price"], 
                 names=price.sort_values(ascending = False).head(20).reset_index()["brand_title"], 
                 title='Market share')
    with col2:
        st.plotly_chart(fig, use_container_width=True) 

    st.title("Status")

    col1, col2 = st.columns([0.7, 0.3])

    price = st.session_state.products_catalog.groupby(["status"])["price"].sum()
    price = price/price.sum()

    products = st.session_state.products_catalog.groupby(["status"])["product_id"].count()
    products = products/products.sum()
    fig = go.Figure(data=[
        go.Bar(name='Price', 
               x=price.sort_values(ascending = False).head(20).reset_index()["status"], 
               y=price.sort_values(ascending = False).head(20).reset_index()["price"]
               ),
        go.Bar(name='Count', 
               x=products.sort_values(ascending = False).head(20).reset_index()["status"], 
               y=products.sort_values(ascending = False).head(20).reset_index()["product_id"])
    ])
    fig.update_layout(title = "Normalized price and count")
    with col1:
        st.plotly_chart(fig, use_container_width=True) 

    fig = px.pie(price, 
                 values=price.sort_values(ascending = False).head(20).reset_index()["price"], 
                 names=price.sort_values(ascending = False).head(20).reset_index()["status"], 
                 title='Market share')
    with col2:
        st.plotly_chart(fig, use_container_width=True) 

    # do median with boxplot
    # do stacked bar with percentage of total count and price

    brands = st.session_state.products_catalog.groupby(["status"])["product_id"].count().sort_values(ascending = False).head(20).reset_index()["status"]

    fig = px.box(st.session_state.products_catalog[st.session_state.products_catalog["status"].isin(brands)], 
                       y="price", 
                       color="status",
                       orientation="v",
                       title= "Status-price boxplot")
    st.plotly_chart(fig, use_container_width=True) 


if __name__ == "__main__":
    main()