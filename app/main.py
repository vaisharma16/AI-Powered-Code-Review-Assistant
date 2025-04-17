import streamlit as st
from review_generator import generate_review
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

st.set_page_config(page_title="AI Code Review Assistant", page_icon="🤖")
st.title("🤖 AI-Powered Code Review Assistant")

uploaded_file = st.file_uploader("Upload your code file (.py)", type="py")


if uploaded_file is not None:
    code = uploaded_file.read().decode("utf-8")
    if len(code) > 1000:
        code = code[:1000]
    st.code(code, language="python")

    st.subheader("📝 Code Review Summary")
    with st.spinner("Analyzing your code..."):
        try:
            review = generate_review(code)
            st.write("Generated Review:", review)  # Debugging line
            if review:
                st.write(review)
            else:
                st.warning("⚠️ No review generated. Try simplifying the code or checking model setup.")
        except Exception as e:
            st.error(f"🚨 Error: {e}")
    st.subheader("📝 Code Review Summary")
    st.text("🧪 Raw Output Below:")
    st.code(review, language="markdown")


