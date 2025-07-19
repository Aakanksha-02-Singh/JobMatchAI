import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("ai_test_recommendation_data.csv")

df['All_Tests'] = df[['Recommended Test 1', 'Recommended Test 2', 'Recommended Test 3']].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(df['Job Description'])

# Streamlit App
st.set_page_config(page_title="JobMatchAI", layout="centered")
st.title("🔍 JobMatchAI")

user_input = st.text_area("Enter the Job Description:", height=200)

if st.button("Recommend Tests"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Transform the input 
        input_vec = vectorizer.transform([user_input])

        # Calculate cosine similarity
        similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()

        # Get top 5 most similar job descriptions
        top_indices = similarities.argsort()[-5:][::-1]

        st.subheader("Top Recommended Tests Based on Similar Job Descriptions:")
        recommended_tests = []
        for idx in top_indices:
            job_title = df.iloc[idx]['Job Title']
            tests = df.iloc[idx][['Recommended Test 1', 'Recommended Test 2', 'Recommended Test 3']].values.tolist()
            recommended_tests.extend(tests)

        
        seen = set()
        final_tests = []
        for test in recommended_tests:
            if test not in seen:
                seen.add(test)
                final_tests.append(test)
            if len(final_tests) == 5:
                break

        for i, test in enumerate(final_tests, start=1):
            st.write(f"{i}. 🧪 {test}")
