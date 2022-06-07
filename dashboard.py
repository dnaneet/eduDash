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
st.set_page_config(page_title='Learning Analytics Dashboard',layout='wide')

# gradebook import + sidebar text + tool selection
# gradebook=st.sidebar.file_uploader(label='Please choose your gradebook CSV file',type=['csv'])
# if gradebook is not None:
#         gb_all=pd.read_csv(gradebook)

st.sidebar.markdown("# Choose your analysis tool")

selection=st.sidebar.radio(label='Tool Choice',options=['Failure Modes','Clustering'])

st.sidebar.markdown("---")
st.sidebar.write("This web-app allows multi-dimensional visualization and analysis of student work-products and gradebook .")
st.sidebar.markdown("###### Created in Python with Streamlit.")
st.sidebar.markdown("###### GUI development and Code adaptation by Ry Swaty.")
st.sidebar.markdown("###### GUI and Code administration by Aneet Narendanath, PhD (C) 2022")
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
    df_gb = pd.read_csv('gradebook2.csv')
    #
    #df_gb["homework"] = df_gb["homework"]*100/200;
    #df_gb["teamwork"] = df_gb["teamwork"]*100/400
    #df_gb["exams"] = df_gb["exams"]*100/400
    #st.write(data_gb.columns)

    #importing data
    st.subheader("3D representation of three categories of grades")
    xdata = st.selectbox('Select your X axis [numeric only]', np.array(df_gb.columns))
    ydata = st.selectbox('Select your Y axis [numeric only]', np.array(df_gb.columns))
    zdata = st.selectbox('Select your Z axis [numeric only]', np.array(df_gb.columns))
    colordata = st.selectbox('Select what you would like to color your data by [numeric only]', np.array(df_gb.columns))
                         
    fig_gb = px.scatter_3d(df_gb, x=xdata, y=ydata, z=zdata, 
                    color=colordata) #letter grade
    fig_gb.update_layout(scene = dict(
                        xaxis = dict(nticks=4, range=[0,110],),
                        yaxis = dict(nticks=4, range=[0,110],),
                        zaxis = dict(nticks=4, range=[0,110],),),
                        margin=dict(r=20, l=10, b=10, t=10),
                        height=800, width=800, title_text="letter grades")
    st.plotly_chart(fig_gb,use_container_width=True)
    
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    pca_scores = pca.fit_transform(df_gb.iloc[:,0:3])
    #st.write(pca.explained_variance_ratio_)  
    transformed = pd.DataFrame(pca_scores)

    pca12 = np.array(transformed)
    
    # Clustering    
    nc = st.slider('Select number of clusters desired',0,10,1)
    st.write("It is suggested that you pick the same number of clusters as you have traditional letter grades.  This will allow you to compare and contrast letter grades with clusters.")
    kmeans = KMeans(n_clusters=nc,random_state=0)
    kmeans.fit(pca12)
    labels_kmean = kmeans.predict(pca12)
    #st.write(labels_kmean)
    
    df_gb["kmeans cluster"] = labels_kmean
    df2 = df_gb.sort_values('kmeans cluster')
    
    fig_clusters = px.scatter_3d(df2.iloc[:,0:10], x='homework', y='teamwork', z='exams', 
                    color='kmeans cluster')
    fig_clusters.update_layout(scene = dict(
        xaxis = dict(nticks=4, range=[0,110],),
                     yaxis = dict(nticks=4, range=[0,110],),
                     zaxis = dict(nticks=4, range=[0,110],),),
                     margin=dict(r=20, l=10, b=10, t=10))
    st.plotly_chart(fig_clusters,use_container_width=True)
    
    df2.to_csv('cluster-data.csv', index=False)
    #st.markdown("The clustered gradebook data is available via [this link](cluster-data.csv).")
    
