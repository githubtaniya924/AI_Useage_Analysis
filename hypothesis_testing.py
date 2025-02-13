import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ai_use_dataset_final.csv")
    return df

df = load_data()

st.write("""
         ## Hypothesis Testing: AI Training vs Work Efficiency Score
         """)
st.divider()

st.write("""
#### Does AI Training Significantly Impact Work Efficiency Scores?
**Columns to Consider:**
- "AI Training Received" (Categorical - Binary)
- "Work Efficiency Score" (Continuous - Scale)
           
We use **Welch's t-test** to compare the mean **Work Efficiency Scores** of two independent groups:  
- **AI Trained Users** (Received AI Training)  
- **Non-AI Trained Users** (Did Not Receive AI Training)  
""")

def t_test(df):

    #-------------------1---------------------#
    # Splitting the data into two groups based on AI Training
    trained = df[df['AI Training Received'] == 'Yes']['Work Efficiency Score']
    not_trained = df[df['AI Training Received'] == 'No']['Work Efficiency Score']
    

    # Performing an independent t-test
    t_stat, p_value = stats.ttest_ind(trained, not_trained, equal_var=False)  # Welch's t-test
    
    # Calculate mean and standard deviation
    trained_mean = trained.mean()
    trained_sd = trained.std()
    not_trained_mean = not_trained.mean()
    not_trained_sd = not_trained.std()

    # 📊 **Table: Mean and Standard Deviation**
    st.write("Mean & Standard Deviation Table")

    stats_df = pd.DataFrame({
        "Group": ["Trained in AI", "Not Trained in AI"],
        "Mean": [trained_mean, not_trained_mean],
        "Standard Deviation": [trained_sd, not_trained_sd]
    })

    st.table(stats_df)


    # 📊 **Generate Boxplot**
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.title("Work Efficiency Score Distribution")
    sns.boxplot(x=df['AI Training Received'], y=df['Work Efficiency Score'], palette=["#E74C3C", "#2E86C1"], ax=ax)
    ax.set_xlabel("AI Training Received", fontsize=10)
    ax.set_ylabel("Work Efficiency Score", fontsize=10)
    ax.set_title("Work Efficiency Score by AI Training Status", fontsize=12)
    st.pyplot(fig)

    
    # Display results in Streamlit
    st.write("""<u><b>By applying Welch-t test, we get the following result:</b></u>""", unsafe_allow_html=True)
    st.write(f"**T-statistic:** {t_stat:.4f}")
    st.write(f"**P-value:** {p_value:.4f}")
    st.write("""
             <i><b>If p-value < 0.05 → Reject H₀ → AI Training significantly impacts Work Efficiency Scores.  
             If p-value ≥ 0.05 → Fail to Reject H₀ → No significant impact.</b></i>  
             """, unsafe_allow_html=True)


    # Interpretation of results
    alpha = 0.05  # Significance level
    if p_value < alpha:
        st.success("Therefore, Reject Null Hypothesis: AI Training **significantly impacts** Work Efficiency Score.")
    else:
        st.warning("Therefore, Fail to Reject Null Hypothesis: AI Training **does not significantly impact** Work Efficiency Score.")
    

# Run the hypothesis test

t_test(df)

st.divider()

    #-------------------2------------------#

st.write("""
### **Does Frequent AI Usage Lead to More Job Promotions or Salary Increases?**
- **Dependent Variable:** Job Promotions / Salary Increases (**Binary: Yes/No**)  
- **Independent Variable:** Frequency of AI Usage (**Ordinal: Never, Rarely, Sometimes, Often, Very Often**)  
- **Statistical Test Used:** **Mann-Whitney U Test**  
  - Since AI usage is ordinal (ranked categories), and job promotions are binary, we use Mann-Whitney U Test to compare distributions.
""")


### **📊 Crosstab Table**
st.write("### Distribution of Job Promotions/Salary Increases Across AI Usage Levels")

# Create Crosstab
crosstab = pd.crosstab(df["Frequency of AI Use"], df["Job Promotions or Salary Increase"], normalize="index") * 100
crosstab_table = pd.crosstab(df["Frequency of AI Use"], df["Job Promotions or Salary Increase"])

# Display Crosstab Table
st.table(crosstab_table)

st.divider()

### **📊 Stacked Bar Chart (vi_bar_stacked_multiple)**
st.write("### **Visualization: AI Usage vs Job Promotions/Salary Increases**")
fig = px.bar(crosstab, 
             x=crosstab.index, 
             y=["Yes", "No"], 
             barmode="stack",
             title="Percentage of Job Promotions/Salary Increases by AI Usage Level",
             labels={"value": "Percentage", "variable": "Promotion/Salary Increase"},
             color_discrete_map={"Yes": "#2E86C1", "No": "#E74C3C"})  # Blue for Yes, Red for No

st.plotly_chart(fig)

st.divider()

### **📊 Mann-Whitney U Test**
st.write("### **Hypothesis Testing: Mann-Whitney U Test**")

# Convert ordinal AI usage into numerical ranks
ordinal_mapping = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very Often": 5}
df["AI Usage Rank"] = df["Frequency of AI Use"].map(ordinal_mapping)

# Split data based on job promotions/salary increases
group_yes = df[df["Job Promotions or Salary Increase"] == "Yes"]["AI Usage Rank"]
group_no = df[df["Job Promotions or Salary Increase"] == "No"]["AI Usage Rank"]

# Perform Mann-Whitney U Test
u_stat, p_value = stats.mannwhitneyu(group_yes, group_no, alternative='greater')

# Display results
st.write(f"**U-Statistic:** {u_stat:.4f}")
st.write(f"**P-value:** {p_value:.4f}")

st.write("""
**Interpretation:**  
- If **p-value < 0.05**, reject the null hypothesis → **Frequent AI users are significantly more likely to receive job promotions or salary increases.**  
- If **p-value ≥ 0.05**, fail to reject the null hypothesis → **No significant difference between frequent and infrequent AI users.**  
""")

# Final Conclusion
alpha = 0.05
if p_value < alpha:
    st.success("Conclusion: Frequent AI Users **significantly** receive more job promotions or salary increases.")
else:
    st.warning("Conclusion: No significant difference in job promotions/salary increases based on AI usage.")

st.divider()

#-------------------3------------------------#

# Section: Description
st.write("""
### **Does Industry Type Affect Perceived Productivity Increase?**
- **Dependent Variable:** Perceived Productivity Increase (**Scale/Continuous**)  
- **Independent Variable:** Industry (**Nominal/Categorical**)  
- **Statistical Test Used:** **Welch ANOVA**  
  - Welch ANOVA is used instead of regular ANOVA because it accounts for unequal variances among groups.
""")

st.divider()

### **📊 Summary Table**
st.write("### **Summary Statistics: Productivity Increase by Industry**")

# Compute summary statistics
summary_table = df.groupby("Industry")["Perceived Increase in Productivity (%)"].agg(["mean", "std", "count"]).reset_index()
summary_table.columns = ["Industry", "Mean Productivity Increase (%)", "Std Dev", "Count"]

# Display Summary Table
st.table(summary_table)

st.divider()

### **📊 Bar Chart**
st.write("### **Visualization: Mean Productivity Increase by Industry**")

fig = px.bar(summary_table, 
             x="Industry", 
             y="Mean Productivity Increase (%)", 
             title="Mean Perceived Productivity Increase by Industry",
             labels={"Mean Productivity Increase (%)": "Perceived Increase in Productivity (%)"},
             color="Industry",
             text="Mean Productivity Increase (%)",
             height=500)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig)

st.divider()

### **📊 Welch ANOVA Test**
st.write("### **Hypothesis Testing: Welch ANOVA**")

# Perform Welch ANOVA Test
industries = df["Industry"].unique()
grouped_data = [df[df["Industry"] == industry]["Perceived Increase in Productivity (%)"].dropna() for industry in industries]
anova_stat, p_value = stats.f_oneway(*grouped_data)  # Welch ANOVA alternative

# Display results
st.write(f"**F-Statistic:** {anova_stat:.4f}")
st.write(f"**P-value:** {p_value:.4f}")

st.write("""
**Interpretation:**  
- If **p-value < 0.05**, reject the null hypothesis → **Industries have significantly different perceived productivity increases.**  
- If **p-value ≥ 0.05**, fail to reject the null hypothesis → **No significant difference across industries.**  
""")

# Final Conclusion
alpha = 0.05
if p_value < alpha:
    st.success("Conclusion: Different industries **significantly** differ in perceived productivity increase.")
else:
    st.warning("Conclusion: No significant difference in perceived productivity increase across industries.")

st.divider()

#-----------------4-------------------------------#

st.write("""
### **Does the Choice of AI Tools Depend on the Purpose of AI Usage?**
- **Variable 1:** AI Tools Used (**Categorical**)  
- **Variable 2:** Purpose of AI Usage (**Categorical**)  
- **Statistical Test Used:** **Chi-Square Test for Independence**  
  - Determines whether the two categorical variables are independent.
""")

st.divider()

### **📊 Contingency Table**
st.write("### **Contingency Table: AI Tools vs Purpose of AI Usage**")

# Create contingency table
contingency_table = pd.crosstab(df["AI Tools Used"], df["Purpose of AI Usage"])
st.table(contingency_table)

st.divider()

### **📊 Stacked Bar Chart for Visualization**
st.write("### **Visualization: AI Tools Usage by Purpose**")

fig = px.bar(df, x="AI Tools Used", color="Purpose of AI Usage", 
             title="Distribution of AI Tools Used for Different Purposes",
             barmode="stack", height=500)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

st.divider()

### **📊 Chi-Square Test for Independence**
st.write("### **Hypothesis Testing: Chi-Square Test**")

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Display results
st.write(f"**Chi-Square Statistic (χ²):** {chi2_stat:.4f}")
st.write(f"**Degrees of Freedom:** {dof}")
st.write(f"**P-value:** {p_value:.4f}")

st.write("""
**Interpretation:**  
- If **p-value < 0.05**, reject H₀ → **AI Tools Used and Purpose of AI Usage are significantly related.**  
- If **p-value ≥ 0.05**, fail to reject H₀ → **No significant relationship between the variables.**  
""")

# Final Conclusion
alpha = 0.05
if p_value < alpha:
    st.success("Conclusion: There is a significant relationship between AI Tools Used and Purpose of AI Usage.")
else:
    st.warning("Conclusion: No significant relationship found between AI Tools Used and Purpose of AI Usage.")

st.divider()