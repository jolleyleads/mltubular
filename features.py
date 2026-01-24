import pandas as pd

def engineer_features(df):
    """
    Feature engineering for ML model.
    - Creates tenure_bucket
    - Converts categorical columns to numeric codes
    """

    # Create tenure_bucket from tenure (adjust bins and labels as needed)
    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(
            df['tenure'], 
            bins=[0, 6, 12, 24, 60], 
            labels=['new', 'short', 'mid', 'long']
        )

    # Automatically convert all object/categorical columns to numeric codes
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = pd.Categorical(df[col]).codes

    return df
