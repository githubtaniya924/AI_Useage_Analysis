import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ai_use_dataset_final.csv")
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset Overview", "Analysis"])

if page == "Home":
    st.title("Welcome to AI Usage Analysis Dashboard")
    st.write("Explore the impact of AI usage on productivity and performance.")

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("Here's a preview of the dataset:")
    st.dataframe(df.head(10))  # Display first 10 rows

    st.write("### Dataset Summary")
    st.write(df.describe())  # Summary statistics

elif page == "Analysis":
    st.sidebar.header("Analysis Options")
    analysis_option = st.sidebar.radio("Select Analysis Type", ["Basic", "Hypothesis Testing", "Predictive Analysis"])

    if analysis_option == "Basic":
        st.title("Basic Analysis & Insights")
        
        # Filters
        st.sidebar.header("Filters")
        User_Type = st.sidebar.selectbox("Select User Type", ["All", "Employee", "Student", "Others"])
        if User_Type != "All":
            df = df[df["User_Type"] == User_Type]

        # Function for various visualizations
        def plot_analysis(df):
            if 'Year' in df.columns:
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
                df = df.dropna(subset=["Year"])
            else:
                st.error("'Year' column not found in the dataset.")
                return

            min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
            selected_year = st.slider("Select Year", min_value=min_year, max_value=max_year, value=max_year)
            df_filtered = df[df["Year"] <= selected_year]

            yearly_trend = df_filtered.groupby("Year").size().reset_index(name="AI Usage Count")
            st.subheader("When Users Started Using AI")
            fig = px.line(yearly_trend, x="Year", y="AI Usage Count",
                          title="Trend of AI Usage Over the Years",
                          markers=True, line_shape="spline")
            st.plotly_chart(fig)

            st.subheader("Distribution of Users by Type")
            User_Type_count = df["User_Type"].value_counts().reset_index()
            User_Type_count.columns = ["User_Type", "Count"]
            User_Type_fig = px.bar(User_Type_count, x="User_Type", y="Count",
                                   title="User Type Distribution",
                                   color="User_Type",
                                   color_discrete_map={"Employee": "#FF5733", "Student": "#33FF57", "Others": "#3357FF"})
            User_Type_fig.update_layout(xaxis_title="User_Type", yaxis_title="Count", xaxis_tickangle=-45)
            st.plotly_chart(User_Type_fig)

            task_time_fig = px.bar(df, x="User_Type", y=["Avg Task Time (Before AI)", "Avg Task Time (After AI)"],
                                   title="Average Task Time Before and After AI Adoption",
                                   labels={"Avg Task Time (Before AI)": "Before AI",
                                           "Avg Task Time (After AI)": "After AI"},
                                   barmode="group")
            st.plotly_chart(task_time_fig)

            df_counts = df["Skill Development Areas"].value_counts().reset_index()
            df_counts.columns = ["Skill Development Areas", "Count"]
            skill_development_fig = px.pie(df_counts, names="Skill Development Areas", values="Count",
                                           hole=0.4,
                                           title="Skill Development Areas Influenced by AI Usage")
            st.plotly_chart(skill_development_fig)

            ai_tools_fig = px.histogram(df, x="Education Level", color="AI Tools Used",
                                        title="AI Tools Usage by Education Level", barmode="stack")
            st.plotly_chart(ai_tools_fig)

            df["User_Type"] = df["User_Type"].astype(str)
            df["Frequency of AI Use"] = df["Frequency of AI Use"].astype(str)
            df = df.dropna(subset=["User_Type", "Frequency of AI Use"])

            plt.figure(figsize=(10, 6))
            sns.countplot(x="Frequency of AI Use", hue="User_Type", data=df, palette="muted")
            plt.title("AI Usage Frequency Among Students and Employees")
            plt.xlabel("Frequency of AI Use")
            plt.ylabel("Count")
            plt.legend(title="User Type")
            st.pyplot(plt)

            st.title("AI Tool Usage Across Different Industries")

            st.write("This visualization shows how AI tools are used in different industries.")

            # Grouping data by industry
            industry_distribution = df.groupby("Industry")["AI Tool Usage"].count().reset_index()
            industry_distribution = industry_distribution.sort_values(by="AI Tool Usage", ascending=False)

            #Bar Chart
            st.subheader("Bar Chart: AI Tool Usage per Industry")
            fig_bar = px.bar(
              industry_distribution,
              x="Industry",
              y="AI Tool Usage",
              title="AI Tool Usage Across Industries",
              labels={"AI Tool Usage": "Number of AI Tools Used", "Industry": "Industry"},
              color="AI Tool Usage",
              color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_bar)

        plot_analysis(df)

    elif analysis_option == "Hypothesis Testing":
        st.title("Hypothesis Testing")
        st.write("Coming soon: Statistical tests to analyze AI's impact on productivity.")

        # Function for Hypothesis testing
        def hypo_testing(df):
            # Splitting the data into two groups based on AI Training
            trained = df[df['AI Training Received'] == 'Yes']['Work Efficiency Score']
            not_trained = df[df['AI Training Received'] == 'No']['Work Efficiency Score']
    
            # Performing an independent t-test
            t_stat, p_value = stats.ttest_ind(trained, not_trained, equal_var=False)  # Welch's t-test (does not assume equal variance)
    
            # Printing the results
            print("Mean Work Efficiency Score for Trained Users:", trained.mean())
            print("Mean Work Efficiency Score for Untrained Users:", not_trained.mean())
            print("T-statistic:", t_stat)
            print("P-value:", p_value)

            # Interpretation
            alpha = 0.05  # Significance level
            if p_value < alpha:
                print("Result: Reject Null Hypothesis - AI Training significantly impacts Work Efficiency Score.")
            else:
                print("Result: Fail to Reject Null Hypothesis - No significant impact of AI Training on Work Efficiency Score.")

        # Running the function
        hypo_testing(df)


    elif analysis_option == "Predictive Analysis":
        st.title("Predictive Analysis")
        st.write("Coming soon: AI-driven predictions on performance and efficiency.")

    

# Basic to Advanced Data Analysis
# What is the distribution of AI tool usage across different industries?
# How does AI training affect the frequency of AI use?
# What is the relationship between AI usage and perceived productivity improvement?
# How do AI-generated content usage percentages vary across different job roles?
# What are the most common challenges faced by users in different industries when integrating AI?
# How does the frequency of AI usage correlate with job promotions or salary increases?
# Is there a trend in AI adoption over the years among students versus professionals?
# Which industries report the highest efficiency improvements due to AI?
# What is the correlation between AI restrictions in companies/universities and willingness to continue AI usage?
# How does AI integration satisfaction differ between students and working professionals?