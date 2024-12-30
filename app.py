import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airline Satisfaction Data Models", page_icon="✈️")

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Prediction Models"])

df = pd.read_csv('train.csv')

if page == "Home":
    st.title("Airline Satisfaction Data Models")
    st.subheader("An interacive app for exploring the Airlines Satisfaction Data!")
    st.write("""
        This app is desinged to allow you to explore the Airline Satisfaction data presented in an interactive way! You get explinations of all the various data points and view 
        visualizations that further explore relationships between the different features. Then you can also dive into the various models made with this data to 
        predict future customer satisfaction and see their effectiveness!
            Use the sidebar to navigate through the sections!
            """)
    st.image('airport.jpg')

elif page == 'Data Overview':
    st.title("🔢 Data Overview")
    st.subheader("About the Data")
    st.write("""In this dataset we are presented with various features that describe the flight experience of a large number of passangers""")

    st.subheader("Data Summary")
    st.write("Select the options below to get some insights into the data!")
    if st.checkbox("Show Data Sample"):
        st.dataframe(df)

    if st.checkbox("Size of the Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Data Dictionary"):
        st.write("An explination of every feature in the dataset")
        st.image('Dictionary.png')

    if st.checkbox("Average of each feature"):
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        sel_col = st.selectbox("Select a numerical column from the Dataframe", num_cols)

        if sel_col:
            avg = round(df[sel_col].mean(),2)
            st.write(f"The average of {sel_col} is: {avg}!") 

elif page == "Exploratory Data Analysis": 
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Select the type of Chart you would like to see:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Heatmaps'])
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
   

    if 'Histograms' in eda_type:
        num_col = df.select_dtypes(include=['number']).columns.tolist()
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Relationships compared to overall Satisfaction")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))


    if 'Heatmaps' in eda_type:
        st.subheader("Heatmaps - Visualizing Correlations")
        if st.checkbox("Correlation of Features"):
            selected_col = st.selectbox("Select the variable you wish to see corelations for:", num_cols)
      
            if selected_col:
                correlation_matrix = df.corr(numeric_only=True)
                col_corr = correlation_matrix[[selected_col]].sort_values(by=selected_col, ascending=False)
            
        
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.heatmap(
                    col_corr,
                    vmin=-1,
                 vmax=1,
                    annot=True,
                    cmap='coolwarm',
                    ax=ax
                )
                st.pyplot(fig)
        
        if st.checkbox("Correlation to Target"):
            st.write("This heat map shows the correlation for each feature compared to overall satisfaction level, this heat map will be used as a basis for selecting features for the upcoming models!")
            st.image('positive.png')
            st.image('negative.png')


elif page == "Prediction Models":
    st.title("Exploring the Models 📊")
    st.subheader("Comparing the Models: KNN / Logistic Regression / Random Forest")
    st.write("Here we will explore the final iteration of each model and see how well they preformed. After several iteration each model preformed best when using every available feature and those models are what is described below.")
    if st.checkbox('Model Scoring Explained'):
        st.write("Each model will be scored based on several metrics described below:")
        st.image('metrics.png')
    if st.checkbox("Confusion Matrix Explained"):
        st.write("Each model will also have an attached Confusion Matrix to break down the results visually. Here is a breif explination:")
        st.image('confusion.png')
    if st.checkbox('Baseline Goal'):
        st.write("Every model must be compared to a baseline to evaluate its effectiveness. The Baseline is simply the accuracy of the model if the most commonly occuring class was assumed for each input.")
        st.write("For this dataset, that would mean assuming that every passenger was 'neutral or dissatisfied'. This would give us a baseline accuracy of 56.658%.")

    model_type = st.multiselect("Select which model you wish to view", ['KNN', 'Logistic Regression', 'Random Forest'])

    if 'KNN' in model_type:
        st.subheader("K-nearest neighbors (kNN)")
        st.write("K-nearest neighbors (kNN) is a machine learning algorithm that uses proximity to classify or predict data points")
        st.subheader("The Scores")
        st.write("Accuracy: 92.44%")
        st.write("Precision: 95.38%")
        st.write("Recall: 88.80%")
        st.write("Specificity: 96.43%")
        st.subheader("Confusion Matrix")
        st.image('knn_con.png')
        st.write("This model did impressive work with a base accuracty of 92.44% Where is seams to struggle is in Recall with only a 88.80%, meaning it had a tendancy for predicting false negatives. This is reflected in the Confusion Matrix showing 1418 False Negatives total.")

    if 'Logistic Regression' in model_type:
        st.subheader("Logistic Regression")
        st.write("Logistic regression is a technique that uses mathematics to find the relationships between two data factors. It then uses this relationship to predict the value of one of those factors based on the other. The prediction usually has a finite number of outcomes, like yes or no.")
        st.subheader("The Scores")
        st.write("Accuracy: 87.47%")
        st.write("Precision: 89.03%")
        st.write("Recall: 85.78%")
        st.write("Specificity: 91.39%")
        st.subheader("Confusion Matrix")
        st.image('log_con.png')
        st.write("This Model did worse overall with every score being lower than the previous. It is not a bad model by any means, beating the baseline accuracy by a large margin, it is just not the right model for this data.")

if 'Random Forest' in model_type:
        st.subheader("Random Forest")
        st.write("A Random Forest is a machine learning algorithm that combines multiple decision trees to make predictions, essentially creating a 'forest' of trees where each tree is built on a slightly different subset of data, resulting in a more robust and accurate prediction than using a single decision tree alone")
        st.write("Accuracy: 96.25%")
        st.write("Precision: 97.29%")
        st.write("Recall: 94.45%")
        st.write("Specificity: 97.91%")
        st.subheader("Confusion Matrix")
        st.image('forest_con.png')
        st.write("This model preformed the best of any model improving upon the already impressive accuracy of the KNN model. Most notably, having a massive improvement in the Recall metric with a 94.45% vs the KNN's 88.80%.")
