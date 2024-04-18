import pandas as pd
import re


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
            text_parts.append(f"{col_name}_{value}")
    return " ".join(text_parts)


def preprocess_pipeline(
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
    preprocessed_data["firstname"] = preprocessed_data["firstname"].str.lower()

    # Jointure avec les données de nom
    final_data = pd.merge(
        preprocessed_data,
        name_data,
        left_on=["firstname"],
        right_on=["firstname"],
        how="left",
    )

    # Suppression des colonnes redondantes et création de la colonne "texte"
    final_data["texte"] = final_data.apply(concatenate_column_names, axis=1)

    return final_data
