import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert labels: "depression" -> 0, "SuicideWatch" -> 1
    df['label'] = df['label'].map({'depression': 0, 'SuicideWatch': 1})
    
    # Drop rows where either 'text' or 'label' columns have missing values
    df = df.dropna(subset=['text', 'label'])
    
    # Explicitly convert 'label' column to integers
    df['label'] = df['label'].astype(int)
    
    return df



def print_label_distribution(df):
    # Calculate and print the percentage of each label
    total = len(df)
    depression_count = sum(df['label'] == 0)
    suicidewatch_count = sum(df['label'] == 1)
    
    print(f"Percentage of 'depression': {depression_count / total * 100:.2f}%")
    print(f"Percentage of 'SuicideWatch': {suicidewatch_count / total * 100:.2f}%")

def split_and_save_datasets(df):
    # Ensure balanced labels in test and dev sets
    # Split the dataset into train and (test + dev) with a ratio of 7:3 first
    df_train, df_test_dev = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    
    # Further split (test + dev) into test and dev with a ratio of 2:1 (overall 2:7:1)
    df_test, df_dev = train_test_split(df_test_dev, test_size=1/3, stratify=df_test_dev['label'], random_state=42)
    
    # Shuffle the datasets
    df_train = shuffle(df_train, random_state=42)
    df_dev = shuffle(df_dev, random_state=42)
    df_test = shuffle(df_test, random_state=42)
    
    # Save to CSV
    df_train.to_csv('train.csv', index=False)
    df_dev.to_csv('dev.csv', index=False)
    df_test.to_csv('test.csv', index=False)

def main():
    # Specify the path to your dataset
    file_path = 'depression_suicidewatch.csv'
    
    # Load and preprocess the dataset
    df = load_dataset(file_path)
    
    # Print the distribution of labels before any splits
    print_label_distribution(df)
    
    # Split the dataset and save to CSV files
    split_and_save_datasets(df)
    
    print("Datasets have been processed and saved. The splits ensure balanced labels in test and dev sets.")

if __name__ == "__main__":
    main()
