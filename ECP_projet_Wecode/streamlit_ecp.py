import streamlit as st
import pickle
from scipy.sparse import load_npz
import numpy as np
import pandas as pd
import pickle

@st.cache_resource  
def load_model():
    with open("collab_best_model_new.pkl", "rb") as f:
        return pickle.load(f)

def load_data(file_path):
    return pd.read_csv(file_path)

similarity_matrix = load_npz('similarity_matrix_npz.npz')
filtered_features = pd.read_csv('base_cont_df.csv', index_col='itemid')

st.title("Système de Recommandation de Produits")
st.header('❇️À propos des Systèmes de Recommandation: ')
st.markdown('Un système de recommandation est une technologie utilisée pour suggérer des articles ou des produits pertinents à un utilisateur en se basant sur ses préférences, son comportement ou des données similaires.')
st.markdown('Filtrage collaboratif : Utilise les comportements des utilisateurs pour recommander des articles.')
                
st.markdown('Similarité : Recommande des produits similaires en fonction de leurs caractéristiques.')
st.sidebar.header("Sélectionnez un une fonctionnalité")
option = st.sidebar.selectbox(
    "Methodes  de recommandation",
    ("SVD", "Similarité", "Hybride")
)
def find_articles(item_id, similarity_matrix, filtered_features):
    if item_id not in filtered_features.index:
        return f"Le produit {item_id} n'existe dans la liste de produits disponible sur la boutique."
    
    item_index = filtered_features.index.get_loc(item_id)
    similar_scores = similarity_matrix[item_index].toarray().flatten()
    similar_indices = np.argsort(-similar_scores)[:15]
    # similar_items = [(filtered_features.index[i], similar_scores[i]) for i in similar_indices]
    similar_items = [
    (filtered_features.index[i], similar_scores[i])
    for i in similar_indices
    if i < len(filtered_features)
]

    return similar_items
if option == "SVD":
    st.write("Nous utilisons un model de filtrage collaboratif. Pour recommander des produits aux users")
    
    file = "features_svd_new.csv"
    if file:
        features = load_data(file)
        model = load_model()

        visitor_ids = features['visitorid'].unique()
        selected_user = st.selectbox("Choisissez un user pour voir ses recommandations :", visitor_ids)
        if st.button("Afficher les recommandations"):
            items = features['itemid'].unique()
            recommendations = []
            for item_id in items:
                pred = model.predict(selected_user, item_id)
                recommendations.append((item_id, pred.est))
            top_10 = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
            st.subheader("Top 10 des produits recommandés :")
            for item_id, score in top_10:
                st.write(f"Produit : {item_id}, Score : {score}")

elif option == "Similarité":
    st.header("Recommandations basées sur la Similarité")
    item_id = st.selectbox("Entrez l'ID du produit pour trouver des articles similaires", filtered_features.index)

    if st.button("Trouver les articles similaires"):
        simil_prod = find_articles(int(item_id), similarity_matrix, filtered_features)
        if isinstance(simil_prod, str):
            st.write(simil_prod)
        else:
            if simil_prod:
                st.write(f"Articles similaires a {item_id} :")
            else:
                st.write(f'Aucun article similaire à {item_id}')
            
            for items, score in simil_prod:

                    st.write(f"item_ID : {items}, Score de Similarité : {score:.4f}")

elif option == "Hybride":
    st.header("Recommandation Hybride")
    files= "features_svd_new.csv"
    features = load_data(files)
    user_id = st.selectbox("Entrez l'ID Utilisateur",features['visitorid'])
    
    visitor_ids_hybrid=features["itemid"]
    item_id = st.selectbox("Entrez l'ID produit", visitor_ids_hybrid)

    if st.button("Recommander"):

        collab_model = load_model()
        svd_prediction = collab_model.predict(int(user_id), int(item_id)).est
        simil_prod = find_articles(int(item_id), similarity_matrix, filtered_features)
        if isinstance(simil_prod, str):
            st.write(simil_prod)
        else:
            hybrid_scores = []
            for article_id, content_score in simil_prod:
                svd_score = collab_model.predict(int(user_id), article_id).est
                hybrid_score = 0.5 * svd_score + 0.5 * content_score
                hybrid_scores.append((article_id, hybrid_score))

            top_10_hybrid = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:10]
            st.subheader("Top 10 des produits recommandés (Hybride) :")
            for article_id, score in top_10_hybrid:
                st.write(f"Item_id : {article_id}, Score Hybride de similarité: {score:.4f}")
