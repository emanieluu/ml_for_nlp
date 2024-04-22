import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def extract_groundtruth(df, column_to_extract):
    # Créer un ensemble vide pour stocker les noms de colonnes uniques
    unique_columns = set()

    # Parcourir chaque ligne de la colonne "groundtruth"
    for entry in df[column_to_extract]:
        # Utiliser une expression régulière pour extraire les noms de colonnes
        matches = re.findall(r"(\w+):", entry)
        unique_columns.update(matches)

    # Créer des dictionnaires vides pour stocker les données extraites
    extracted_data = {col: [] for col in unique_columns}

    # Parcourir chaque ligne de la colonne "groundtruth" et extraire les données
    for entry in df[column_to_extract]:
        matches = re.findall(r"(\w+): (\w+)", entry)
        entry_data = {col: None for col in unique_columns}
        for match in matches:
            col_name, value = match
            entry_data[col_name] = value
        for col, value in entry_data.items():
            extracted_data[col].append(value)

    # Créer un DataFrame avec les données extraites
    new_df = pd.DataFrame(extracted_data)

    return new_df


def concatenate_column_names(row):
    text_parts = []
    for col_name, value in row.iteritems():
        if pd.notna(value) and value != "None":
            text_parts.append(f"{col_name} : {value}")
    return " ".join(text_parts)


def cleaning_pipeline(
    df, name_data, column_to_extract: str, column_to_drop: list = []
):

    data = df.copy()
    data = data.drop(column_to_drop, axis=1)
    # Extraction des données groundtruth
    new_dataframe = extract_groundtruth(df, column_to_extract)

    # Ajout des colonnes extraites à notre DataFrame principal
    for column in new_dataframe.columns:
        data[column] = new_dataframe[column]

    # Suppression des colonnes non nécessaires
    preprocessed_data = data.drop([column_to_extract], axis=1)

    # Conversion des prénoms en minuscules
    preprocessed_data["prénom"] = preprocessed_data["prénom"].str.lower()

    # Jointure avec les données de nom
    final_data = pd.merge(
        preprocessed_data,
        name_data,
        left_on=["prénom"],
        right_on=["prénom"],
        how="left",
    )

    # Suppression des colonnes redondantes et création de la colonne "texte"
    # final_data["texte"] = final_data.apply(concatenate_column_names, axis=1)
    final_data = final_data.replace({None: np.nan})

    return final_data


def calculate_scores(y_true, y_pred, label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=label)
    recall = recall_score(y_true, y_pred, pos_label=label)
    f1 = f1_score(y_true, y_pred, pos_label=label)

    scores_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    return pd.DataFrame(scores_dict, index=[label])
