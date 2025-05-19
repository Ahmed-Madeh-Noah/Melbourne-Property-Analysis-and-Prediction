import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/ANALYSED_Melbourne_Housing_Market.csv")

st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Plot Settings")
x_col = st.selectbox("X-axis column", df.columns)
y_col = st.selectbox("Y-axis column (optional)", [None] + list(df.columns))
hue_col = st.selectbox("Hue column (optional)", [None] + list(df.columns))
plot_type = st.selectbox("Plot type", [
    "scatterplot", "lineplot", "boxplot", "barplot", "histplot", "violinplot"
])

if st.button("Generate Plot"):
    try:
        fig = plt.figure()
        plot_func = getattr(sns, plot_type)
        plot_func(
            data=df,
            x=x_col,
            y=y_col if y_col else None,
            hue=hue_col if hue_col else None
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Plot Error: {e}")
