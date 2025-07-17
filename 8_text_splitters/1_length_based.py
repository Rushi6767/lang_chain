from langchain.text_splitter import CharacterTextSplitter

text = """
E-commerce has evolved significantly over the past decade,
with machine learning transforming how products are recommended.
Personalization is now at the core of the user experience.
Our project integrates ML to provide smart product suggestions.
We analyze user behavior, purchase history, and preferences.
The site supports login, search, and secure checkout features.
Each product has detailed descriptions and customer reviews.
The ML model improves continuously with new user data.
We use collaborative filtering and NLP for recommendations.
Admins can add products, manage inventory, and view analytics.
The UI is responsive, modern, and built with React.
The backend is powered by Django and PostgreSQL.
Stripe is used for secure and reliable payment processing.
User feedback is collected to enhance the recommendation engine.
Overall, this platform offers a seamless and intelligent shopping experience.
"""

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)

print(result)

# print(result[1].page_content)