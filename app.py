import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="EC2 & S3 Usage EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# TITLE
# ========================================
st.title("EC2 & S3 Usage: Exploratory Data Analysis")
st.markdown("**INFO 49971 – Cloud Economics | Week 9 Activity**")

# ========================================
# DATA LOADING
# ========================================
data_folder = "data"

if not os.path.exists(data_folder):
    st.error(f"Folder '{data_folder}' not found. Please create it and add the CSV files.")
    st.stop()

# --- Load EC2 Data ---
ec2_path = os.path.join(data_folder, "aws_resources_compute.csv")
if not os.path.exists(ec2_path):
    st.error(f"EC2 file not found: {ec2_path}")
    st.stop()

df_ec2 = pd.read_csv(ec2_path, parse_dates=["CreationDate"], low_memory=False)

# --- Load S3 Data ---
s3_path = os.path.join(data_folder, "aws_resources_S3.csv")
if not os.path.exists(s3_path):
    st.error(f"S3 file not found: {s3_path}")
    st.stop()

df_s3 = pd.read_csv(s3_path, parse_dates=["CreationDate"])

st.success(f"Loaded EC2 ({len(df_ec2):,} rows) and S3 ({len(df_s3):,} rows) datasets.")

# ========================================
# DATA CLEANING
# ========================================
df_ec2['CPUUtilization'] = df_ec2['CPUUtilization'].fillna(df_ec2['CPUUtilization'].median())
df_ec2['MemoryUtilization'] = df_ec2['MemoryUtilization'].fillna(df_ec2['MemoryUtilization'].median())
df_ec2['CostUSD'] = df_ec2['CostUSD'].fillna(0)

df_s3['CostUSD'] = df_s3['CostUSD'].fillna(0)
df_s3['TotalSizeGB'] = df_s3['TotalSizeGB'].fillna(0)

# Remove outliers
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

df_ec2_clean = remove_outliers(df_ec2, 'CostUSD')
df_s3_clean = remove_outliers(df_s3, 'TotalSizeGB')

# ========================================
# SIDEBAR FILTERS
# ========================================
st.sidebar.header("Filters")

ec2_regions = st.sidebar.multiselect("EC2 Regions", df_ec2["Region"].unique(), default=df_ec2["Region"].unique())
ec2_states = st.sidebar.multiselect("EC2 State", df_ec2["State"].unique(), default=["running"])
ec2_types = st.sidebar.multiselect("Instance Type", df_ec2["InstanceType"].unique())

s3_regions = st.sidebar.multiselect("S3 Regions", df_s3["Region"].unique(), default=df_s3["Region"].unique())
s3_classes = st.sidebar.multiselect("Storage Class", df_s3["StorageClass"].unique())

# Apply Filters
filtered_ec2 = df_ec2[
    df_ec2["Region"].isin(ec2_regions) &
    df_ec2["State"].isin(ec2_states) &
    (df_ec2["InstanceType"].isin(ec2_types) if ec2_types else True)
].copy()

filtered_s3 = df_s3[
    df_s3["Region"].isin(s3_regions) &
    (df_s3["StorageClass"].isin(s3_classes) if s3_classes else True)
].copy()

# ========================================
# COMPUTATIONS
# ========================================
top5_ec2 = filtered_ec2.nlargest(5, 'CostUSD')[['ResourceId', 'InstanceType', 'Region', 'CostUSD']]
top5_s3 = filtered_s3.nlargest(5, 'TotalSizeGB')[['BucketName', 'Region', 'TotalSizeGB', 'CostUSD']]

avg_cost_region = filtered_ec2.groupby('Region')['CostUSD'].mean().reset_index()
total_storage_region = filtered_s3.groupby('Region')['TotalSizeGB'].sum().reset_index()

# ========================================
# DATA-DRIVEN OPTIMIZATION INSIGHTS
# ========================================
# 1. Underutilized EC2: CPU < 20%
underutilized_ec2 = filtered_ec2[
    (filtered_ec2['CPUUtilization'] < 20) &
    (filtered_ec2['State'] == 'running')
].copy()
underutilized_ec2['SavingsPerHour'] = underutilized_ec2['CostUSD'] * 0.5  # Assume 50% savings by downsizing
underutilized_ec2 = underutilized_ec2.sort_values('SavingsPerHour', ascending=False).head(10)

# 2. Expensive S3 STANDARD buckets (candidates for IA/Glacier)
expensive_s3 = filtered_s3[
    (filtered_s3['StorageClass'] == 'STANDARD') &
    (filtered_s3['TotalSizeGB'] > 100)
].copy()
expensive_s3['MonthlySavings'] = expensive_s3['TotalSizeGB'] * 0.01  # ~$0.01/GB savings moving to IA
expensive_s3 = expensive_s3.sort_values('MonthlySavings', ascending=False).head(10)

# ========================================
# KPI CARDS
# ========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("EC2 Instances", len(filtered_ec2))
col2.metric("S3 Buckets", len(filtered_s3))
col3.metric("Total EC2 Cost", f"${filtered_ec2['CostUSD'].sum():.2f}")
col4.metric("Total S3 Storage (GB)", f"{filtered_s3['TotalSizeGB'].sum():.1f}")

# ========================================
# TABS
# ========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EC2 Analysis", "S3 Analysis", "Top Items", "Optimization"])

# ---------- TAB 1: Overview ----------
with tab1:
    st.subheader("Dataset Info")
    colA, colB = st.columns(2)
    with colA:
        st.write("**EC2 Dataset**")
        st.write(df_ec2.info())
        st.write(df_ec2.describe())
    with colB:
        st.write("**S3 Dataset**")
        st.write(df_s3.info())
        st.write(df_s3.describe())

    st.write("**Missing Values**")
    colC, colD = st.columns(2)
    with colC:
        st.write("EC2 Missing:")
        st.write(df_ec2.isna().sum())
    with colD:
        st.write("S3 Missing:")
        st.write(df_s3.isna().sum())

# ---------- TAB 2: EC2 Analysis ----------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**CPU Utilization Distribution**")
        fig_cpu = px.histogram(filtered_ec2, x="CPUUtilization", nbins=30, title="EC2 CPU Usage (%)", color="Region")
        st.plotly_chart(fig_cpu, use_container_width=True)

    with col2:
        st.write("**CPU vs Cost**")
        fig_scatter = px.scatter(
            filtered_ec2, x="CPUUtilization", y="CostUSD",
            color="InstanceType", size="MemoryUtilization",
            hover_data=["ResourceId", "Region"], title="CPU vs Cost"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("**Average Cost per Region**")
    fig_bar = px.bar(avg_cost_region, x="Region", y="CostUSD", title="Avg Cost by Region", text="CostUSD")
    fig_bar.update_traces(texttemplate='%{text:.3f}')
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------- TAB 3: S3 Analysis ----------
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Total Storage by Region**")
        fig_storage = px.bar(total_storage_region, x="Region", y="TotalSizeGB", title="Total S3 Storage (GB)", text="TotalSizeGB")
        fig_storage.update_traces(texttemplate='%{text:.1f}')
        st.plotly_chart(fig_storage, use_container_width=True)

    with col2:
        st.write("**Storage vs Cost**")
        fig_s3_scatter = px.scatter(
            filtered_s3, x="TotalSizeGB", y="CostUSD",
            color="StorageClass", size="ObjectCount",
            hover_data=["BucketName", "Region"], title="Storage vs Cost"
        )
        st.plotly_chart(fig_s3_scatter, use_container_width=True)

# ---------- TAB 4: Top Items ----------
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 5 Most Expensive EC2 Instances**")
        st.dataframe(top5_ec2.reset_index(drop=True), use_container_width=True)
    with col2:
        st.write("**Top 5 Largest S3 Buckets**")
        st.dataframe(top5_s3.reset_index(drop=True), use_container_width=True)

# ---------- TAB 5: DATA-DRIVEN OPTIMIZATION ----------
with tab5:
    st.markdown("### Optimization Recommendations (Based on Your Data)")

    # EC2: Underutilized Instances
    st.markdown("#### **EC2: Downsize Underused Servers**")
    if not underutilized_ec2.empty:
        total_savings_hr = underutilized_ec2['SavingsPerHour'].sum()
        st.metric("**Total Potential Hourly Savings**", f"${total_savings_hr:.2f}")
        st.write("**Top 10 underutilized instances (CPU < 20%)**")
        st.dataframe(
            underutilized_ec2[['ResourceId', 'InstanceType', 'Region', 'CPUUtilization', 'CostUSD', 'SavingsPerHour']]
            .round(2),
            use_container_width=True
        )
        st.info("**Action**: Downsize `m5.large` → `t3.medium`, or use Spot Instances.")
    else:
        st.success("No underutilized instances found!")

    st.markdown("---")

    # S3: Move to Cheaper Storage
    st.markdown("#### **S3: Move Big Buckets to Cheaper Storage**")
    if not expensive_s3.empty:
        total_savings_mo = expensive_s3['MonthlySavings'].sum()
        st.metric("**Total Potential Monthly Savings**", f"${total_savings_mo:.2f}")
        st.write("**Top 10 large STANDARD buckets (>100 GB)**")
        st.dataframe(
            expensive_s3[['BucketName', 'Region', 'TotalSizeGB', 'CostUSD', 'MonthlySavings']]
            .round(2),
            use_container_width=True
        )
        st.info("**Action**: Move to `STANDARD_IA` or `GLACIER` → Save ~50% on storage.")
    else:
        st.success("No expensive STANDARD buckets found!")

# ========================================
# DOWNLOAD BUTTONS
# ========================================
st.sidebar.markdown("---")
st.sidebar.download_button("Download Filtered EC2 Data", filtered_ec2.to_csv(index=False).encode(), "filtered_ec2.csv", "text/csv")
st.sidebar.download_button("Download Filtered S3 Data", filtered_s3.to_csv(index=False).encode(), "filtered_s3.csv", "text/csv")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("Built with Streamlit • Fall 2025 • Sheridan College")