import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    CamembertTokenizer,
    CamembertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW

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


def train_model(
    X_train,
    X_val,
    X_test,
    train_targets,
    val_targets,
    test_targets,
    model_type="bert",
    num_epochs=4,
    learning_rate=2e-5,
    batch_size=16,
):

    # Sélection du tokenizer et du modèle en fonction du type spécifié
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
    elif model_type == "camembert":
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        model = CamembertForSequenceClassification.from_pretrained(
            "camembert-base", num_labels=2
        )
    elif model_type == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
    else:
        raise ValueError(
            "Invalid model type. Choose 'bert', 'camembert', or 'distilbert'."
        )

    # Tokenisation des données
    train_encodings = tokenizer(
        X_train["texte"].tolist(), truncation=True, padding=True
    )
    val_encodings = tokenizer(
        X_val["texte"].tolist(), truncation=True, padding=True
    )
    test_encodings = tokenizer(
        X_test["texte"].tolist(), truncation=True, padding=True
    )

    # Création des datasets PyTorch
    train_dataset = TensorDataset(
        torch.tensor(train_encodings["input_ids"]),
        torch.tensor(train_encodings["attention_mask"]),
        torch.tensor(train_targets),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_encodings["input_ids"]),
        torch.tensor(val_encodings["attention_mask"]),
        torch.tensor(val_targets),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_encodings["input_ids"]),
        torch.tensor(test_encodings["attention_mask"]),
        torch.tensor(test_targets),
    )

    # Création des dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Configuration de l'optimiseur
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Entraînement du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(
                t.to(device) for t in batch
            )

            optimizer.zero_grad()
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}: Average training loss = {avg_train_loss:.4f}"
        )

        # Évaluation du modèle sur l'ensemble de validation à chaque epoch
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = tuple(
                    t.to(device) for t in batch
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch + 1}: Validation accuracy = {val_accuracy:.4f}")

    # Évaluation finale du modèle sur l'ensemble de test
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = tuple(
                t.to(device) for t in batch
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)

    print(f"Final Test accuracy: {test_accuracy:.4f}")
    print(f"Final Test F1 score: {test_f1:.4f}")
