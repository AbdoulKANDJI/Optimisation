import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Fonction principale
def main():
    st.title("Optimisation d'un modèle de classification Decision Tree")
    st.write("Cette application permet de comparer un modèle sans optimisation avec un modèle optimisé.")

    # Étape 1 : Charger le fichier CSV
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Fichier chargé avec succès !")
            st.write(f"Le fichier contient {data.shape[0]} lignes et {data.shape[1]} colonnes.")
            st.write(data.head())
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return

        # Vérifier la colonne 'legitimate'
        if 'legitimate' not in data.columns:
            st.error("La colonne 'legitimate' est absente du fichier. Vérifiez vos données.")
            return

        # Étape 2 : Préparer les données
        X = data.drop('legitimate', axis=1)
        y = data['legitimate']

        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modèle sans optimisation
        st.subheader("Modèle sans optimisation")
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy (sans optimisation) : {accuracy:.2f}")
        st.text("Classification Report :")
        st.text(classification_report(y_test, y_pred))

        # Matrice de confusion
        st.write("Matrice de confusion (sans optimisation) :")
        cm_base = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm_base).plot(ax=ax, colorbar=False)
        st.pyplot(fig)

        # Optimisation avec RandomizedSearchCV
        st.subheader("Optimisation avec RandomizedSearchCV")
        param_dist = {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 20],
            'criterion': ['gini', 'entropy']
        }

        random_search = RandomizedSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=50,
            scoring='f1',
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        # Meilleurs hyperparamètres
        best_params = random_search.best_params_
        st.write("Meilleurs hyperparamètres :")
        st.json(best_params)

        # Modèle optimisé
        best_model = random_search.best_estimator_
        y_pred_optimized = best_model.predict(X_test)

        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        st.write(f"Accuracy (avec optimisation) : {accuracy_optimized:.2f}")
        st.text("Classification Report (optimisé) :")
        st.text(classification_report(y_test, y_pred_optimized))

        # Matrice de confusion optimisée
        st.write("Matrice de confusion (optimisé) :")
        cm_optimized = confusion_matrix(y_test, y_pred_optimized)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm_optimized).plot(ax=ax, colorbar=False)
        st.pyplot(fig)

        # Comparaison des métriques
        st.subheader("Comparaison des métriques")
        report_base = classification_report(y_test, y_pred, output_dict=True)['weighted avg']
        report_optimized = classification_report(y_test, y_pred_optimized, output_dict=True)['weighted avg']

        comparison = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Modèle sans optimisation": [
                report_base["precision"], report_base["recall"], report_base["f1-score"]
            ],
            "Modèle avec optimisation": [
                report_optimized["precision"], report_optimized["recall"], report_optimized["f1-score"]
            ],
        })

        st.table(comparison)

if __name__ == "__main__":
    main()
