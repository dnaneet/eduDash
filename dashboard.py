import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# machine learning functions/packages
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# run from anywhere with: streamlit run "g:/My Drive/PythonSummerProject2022_DrN/Code/Dashboard.py"
# create requirements file with: pipreqs "G:\My Drive\PythonSummerProject2022_DrN\Code" --force

# tab title and page config
st.set_page_config(page_title='Gradebook Explorer',layout='wide')

# gradebook import + sidebar text + tool selection
# gradebook=st.sidebar.file_uploader(label='Please choose your gradebook CSV file',type=['csv'])
# if gradebook is not None:
#         gb_all=pd.read_csv(gradebook)

st.sidebar.markdown("# Use the radio buttons to choose your app")

selection=st.sidebar.radio(label='Tool Choice',options=['Failure Modes','Clustering'])

st.sidebar.markdown("---")
st.sidebar.write("This web-app provides professors a deepened level of data discovery.")
st.sidebar.markdown("###### Created in Python with Streamlit.")
st.sidebar.markdown("###### Created by Ry Swaty in 2022.")
st.sidebar.markdown("---")

# defining universally used values
# cols=list(gb_all.columns)
# cols_new=cols
# cols_new.remove('Student')

#data_csv = pd.read_csv(url="https://raw.githubusercontent.com/dnaneet/eduDash/main/dummy_quiz.csv")
# first tab code
if selection=='Failure Modes':
    st.title('Failure Mode Exploration')

    #importing data
    #data_csv = pd.read_csv(url="https://raw.githubusercontent.com/dnaneet/eduDash/main/dummy_quiz.csv")
    data_csv=st.file_uploader(label='Please choose your assignemnt gradebook',type=['csv'])
    if data_csv is not None:
        data_assign=pd.read_csv(data_csv)

    cols=list(data_assign.columns)

    #user inputs
    data_x=st.selectbox(label='Please select the column that has student names',options=cols)
    data_y=st.selectbox(label='Please select the question you with to view',options=cols)
    coloring=st.selectbox(label='Choose assignment coloring scheme',options=cols)
    
    # plotting
    fig_scatter=px.scatter(data_frame=data_assign,x=data_assign[data_x],y=data_assign[data_y],
    color=coloring,
    title='<b>Student scores',
    height=700)
    
    # figure formating
    fig_scatter.update_layout(title_font_size=20,title_x=0.5)
    fig_scatter.update_xaxes(title_font_size=15,tickfont=dict(size=15),title='X Axis')
    fig_scatter.update_yaxes(title_font_size=15,tickfont=dict(size=15),title='Y Axis')
    fig_scatter.update_traces(marker_size=20)

    # showing plot in streamlit
    st.plotly_chart(fig_scatter,use_container_width=True)

# second tab code
elif selection=='Clustering':
    st.title('3D Plotting and Clustering Analysis')
    height_figs=500

    #importing data
    data_g = pd.read_csv('gradebook.csv')
    # scaling data, this part will only work if the columns listed exist
    data_gb["homework"] = data_gb["homework"]*100/200;
    data_gb["teamwork"] = data_gb["teamwork"]*100/400
    data_gb["exams"] = data_gb["exams"]*100/400

    cols=list(data_gb.columns)
    #st.write('Shown below is your gradebook')
    #st.write(gb_all)
    #st.markdown('### trial')
    
