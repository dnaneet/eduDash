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
    data_csv=st.file_uploader(label='Please choose your gradebook',type=['csv'])
    if data_csv is not None:
        data_gb=pd.read_csv(data_csv)

    # scaling data, this part will only work if the columns listed exist
    data_gb["homework"] = data_gb["homework"]*100/200;
    data_gb["teamwork"] = data_gb["teamwork"]*100/400
    data_gb["exams"] = data_gb["exams"]*100/400

    cols=list(data_gb.columns)
    #st.write('Shown below is your gradebook')
    #st.write(gb_all)
    #st.markdown('### trial')
    student=st.selectbox(label='Please select the column that has student names',options=cols)
    data_x=st.selectbox(label='Please select your x-axis assignement',options=cols)
    data_y=st.selectbox(label='Please select your y-axis assignement',options=cols)
    data_z=st.selectbox(label='Please select your z-axis assignement',options=cols)
    coloring=st.selectbox(label='Choose assignment coloring scheme (Letter Grade)',options=cols)

    # plotting w/o machine learning
    st.markdown('## Normal Grouping')
    fig_3Dscatter=px.scatter_3d(data_frame=data_gb,x=data_gb[data_x],y=data_gb[data_y],z=data_gb[data_z],
    color=coloring,
    hover_data=[student,coloring],
    height=height_figs)

    # figure formating
    fig_3Dscatter.update_layout(scene = dict(
        xaxis = dict(nticks=10, range=[0,110],),
        yaxis = dict(nticks=10, range=[0,110],),
        zaxis = dict(nticks=10, range=[0,110],),),
        margin=dict(r=20, l=10, b=10, t=10))
    
    fig_3Dscatter.update_traces(marker_size=7)

    # showing in streamlit
    st.plotly_chart(fig_3Dscatter,use_container_width=True)


    # machine learning portion
    # data definition
    data_ml=data_gb

    # column position identification 
    pos_x=cols.index(data_x)
    pos_y=cols.index(data_y)
    pos_z=cols.index(data_z)

    # principal component analysis
    pca_dim=sklearnPCA(n_components=2)
    pca_scores=pca_dim.fit_transform(data_ml.iloc[:,[pos_x,pos_y,pos_z]])
    pca_scores_tf=pd.DataFrame(pca_scores)
    pca_12=np.array(pca_scores_tf) 

    # kmeans clustering
    nclus=len(pd.unique(data_ml[coloring])) #shows number of unique grades available in the dataset, useful for clustering
    kmeans=KMeans(n_clusters=nclus,random_state=0)
    kmeans.fit(pca_12)
    kmeans_labels=kmeans.predict(pca_12)

    data_ml['kmeans cluster']=kmeans_labels #creating a new columns that has the cluster
    #data_ml=data_ml.sort_values('kmeans cluster')

    # plotting
    st.markdown('## Plotting with kmeans clustering application')
    fig_3DScatter_ml=px.scatter_3d(data_frame=data_ml,x=data_x,y=data_y,z=data_z,
    color='kmeans cluster',
    hover_data=[student,coloring],
    height=height_figs)

    fig_3DScatter_ml.update_layout(scene = dict(
        xaxis = dict(nticks=10, range=[0,110],),
        yaxis = dict(nticks=10, range=[0,110],),
        zaxis = dict(nticks=10, range=[0,110],),),
        margin=dict(r=20, l=10, b=10, t=10))

    # Showing in Streamlit
    st.plotly_chart(fig_3DScatter_ml,use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: black;'>*Number of clusters is equivalent to number of unique grades present*", unsafe_allow_html=True)
    
