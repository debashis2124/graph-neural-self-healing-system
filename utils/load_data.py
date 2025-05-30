import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_and_preprocess_data(filepath, sheet_name='OrderList', smote_ratio=0.5):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df['Label'] = df['Ship Late Day count'].apply(lambda x: 1 if x > 3 else 0)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Hour'] = df['Order Date'].dt.hour
    df['Day'] = df['Order Date'].dt.day
    df['Weekday'] = df['Order Date'].dt.weekday
    df.ffill(inplace=True)

    categorical_cols = ['Origin Port', 'Carrier', 'Service Level', 'Customer', 'Product ID', 'Plant Code', 'Destination Port']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    selected_features = [
        'Order ID', 'Origin Port', 'Carrier', 'TPT', 'Service Level',
        'Ship ahead day count', 'Ship Late Day count', 'Customer',
        'Product ID', 'Plant Code', 'Destination Port',
        'Unit quantity', 'Weight', 'Hour', 'Day', 'Weekday'
    ]
    scaler = StandardScaler()
    features = scaler.fit_transform(df[selected_features])
    labels = df['Label']

    data_counts = Counter(labels)
    minority_class = min(data_counts, key=data_counts.get)
    k_neighbors = min(5, data_counts[minority_class] - 1) if data_counts[minority_class] > 1 else 1
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=k_neighbors)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    return features_resampled, labels_resampled
