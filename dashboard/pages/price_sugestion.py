import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import subprocess
from wordcloud import WordCloud

from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go

def plotly_wordcloud(text):
    wc = WordCloud(stopwords = set(STOPWORDS),
                   max_words = 200,
                   max_font_size = 100)
    wc.generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(int(i*100))
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       hoverinfo='text',
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',  
                       text=word_list
                      )
    
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

def load_labels():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    sql_query = f"SELECT DISTINCT catalog_id, brand_title FROM public.products_catalog GROUP BY brand_title, catalog_id HAVING COUNT(product_id) > 300"
    df = pd.read_sql(sql_query, engine)
    return (df)   

# Load a sample dataset
def load_data(brand, catalog):
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    sql_query = f"SELECT * FROM public.products_catalog WHERE brand_title = '{brand}' and catalog_id = '{catalog}' ORDER BY date DESC"
    df = pd.read_sql(sql_query, engine)
    return (df)

# Main function
def main():
    st.set_page_config(layout="wide")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Price Sugestion")

    st.session_state.labels = load_labels()

    catalog = st.selectbox(
        'Select the product catalog.',
        options = st.session_state.labels.catalog_id.unique()
    )

    brand = st.selectbox(
        'Select the brand.',
        options = st.session_state.labels[st.session_state.labels["catalog_id"] == catalog]["brand_title"] #.unique()
    )

    st.session_state.products_catalog = load_data(brand = brand, 
                                                  catalog = catalog)

    # Load data
    global products_catalog
    if 'products_catalog' not in st.session_state:
        st.session_state.products_catalog = load_data(brand = brand, catalog = catalog)
        # remove outliers for viz purposes
        q_high = st.session_state.products_catalog["price"].quantile(0.85)
        q_low = st.session_state.products_catalog["price"].quantile(0.05)
        st.session_state.products_catalog = st.session_state.products_catalog[(st.session_state.products_catalog["price"] < q_high) & 
                                                                              (st.session_state.products_catalog["price"] > q_low)]

    cols = st.columns([0.4, 0.4])
    with cols[1]:
        plt.subplots(figsize = (8,8))
        wordcloud = WordCloud (
            background_color = 'white',
                ).generate(' '.join(st.session_state.products_catalog["title"]))
        plt.imshow(wordcloud, interpolation='bilinear')
        st.pyplot(use_container_width= True)        

    with cols[0]: 
        #fig = plotly_wordcloud(st.session_state.products_catalog["title"].str.cat(sep=', '))  
        #st.plotly_chart(fig)  
        fig = px.pie(st.session_state.products_catalog[["status", "product_id"]], 
                    values="product_id", 
                    names="status", 
                    title='Products by status')
        st.plotly_chart(fig)

    fig = px.histogram(st.session_state.products_catalog, x="price", marginal="box", barmode= "overlay", facet_col="status",
                       category_orders={"status":["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]})
    fig.update_xaxes(range = [0,100])
    fig.update_yaxes(range = [0,100])
    st.plotly_chart(fig, use_container_width=True)  

    import numpy as np
    from scipy.stats import bootstrap

    def calculate_quantiles(data):
        return np.percentile(data, [25, 50, 95])
    
    cols = st.columns(5)
    for i, index in enumerate(["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]):
        __ = st.session_state.products_catalog[st.session_state.products_catalog["status"] == index]
        bootstrap_results = bootstrap((__["price"], ), 
                                    statistic=calculate_quantiles,
                                    method = "basic")
        
        df_confidence_interval = pd.DataFrame({
            'low': bootstrap_results.confidence_interval.low,
            'high': bootstrap_results.confidence_interval.high
        },
        index =  ["Q25%", "Q50%", "Q95%"])
        with cols[i]:
            st.write(index, __["price"].count())
            st.table(df_confidence_interval)

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import statsmodels.formula.api as smf

    #manova = MANOVA(endog=data[["brand_title", "status", "size_title", "catalog_id"]], 
    #                exog=data["price"].astype(float))
    #print(manova.mv_test())

    #topbrands = data['brand_title'].value_counts().nlargest(30)
    #filter = topbrands.index.tolist() 
    #data = data[data['brand_title'].isin(filter)]

    tukey_result = pairwise_tukeyhsd(endog=st.session_state.products_catalog['price'], 
                                     groups=st.session_state.products_catalog['status'], 
                                     alpha=0.05)
    for i in ["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]:
        __ = st.session_state.products_catalog[st.session_state.products_catalog["status"] == i]
        if __["price"].count() < 30:
            st.write(f"Carefull in status {i}, not enough samples.")
            
    summ = tukey_result.summary()
    df = pd.DataFrame(summ, 
                      columns = ["group1", "group2", "mean(g2-g1)", "p-value", "lower", "upper", "reject H0"])
    df = df.drop(df.index[0], axis = 0)
    st.subheader("Tukeys HSD")
    st.table(df)

    res = smf.ols(formula="price ~ C(size_title)", 
                  data=st.session_state.products_catalog).fit()
    print(res.params)

    res = smf.ols(formula="price ~ C(status)", 
                  data=st.session_state.products_catalog).fit()
    print(res.params)


main()