import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd


def main():

    # Set up Streamlit interface
    st.set_page_config(
        page_title="📈 Interactive Visualization Tool", page_icon="📈", layout="wide"
    )

    st.header("📈 Interactive Visualization Tool")
    st.write("### Welcome to interactive visualization tool. Please enjoy !")

    # Upload csv files
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here", type="csv")

    # Read csv file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your uploaded data: ", st.session_state.df.head())


    # Render pygwalker
    if st.session_state.get("df") is not None:
        pyg_app = StreamlitRenderer(st.session_state.df)
        pyg_app.explorer()

    else:
        st.info("Please upload a dataset to begin using the interactive visualization tools")


if __name__ == "__main__":
    main()
