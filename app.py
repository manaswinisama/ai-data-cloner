import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

st.title("🧪 Synthetic Data Generator with SDV")
st.write("Upload your CSV file and generate synthetic data using the FAST_ML preset.")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("### 📊 Original Data Preview")
    st.dataframe(data.head())

    st.markdown("### 📈 Original Data Correlation Heatmap")
    if data.select_dtypes(include='number').shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    st.subheader("🧠 Generating Synthetic Data with `FAST_ML`")
    try:
        model = SingleTablePreset(metadata=metadata, name="FAST_ML")
        model.fit(data)
        synthetic_data = model.sample(num_rows=len(data))

        st.success("✅ Synthetic data generated successfully!")

        st.markdown("### 🔹 Preview of Synthetic Data")
        st.dataframe(synthetic_data.head())

        if synthetic_data.select_dtypes(include='number').shape[1] >= 2:
            st.markdown("### 🔸 Correlation Heatmap (Synthetic Data)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(synthetic_data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Download CSV
        csv = synthetic_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="synthetic_data_FAST_ML.csv">📥 Download Synthetic CSV</a>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Failed to generate synthetic data: {str(e)}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
