import pandas as pd
import openai
import argparse
import random
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Replace with your actual OpenAI API key with openai.api_key = 'your own key'


def get_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def few_shot_prompt(text, few_shot_examples=None):
    prompt = (
        "Your task is to identify whether the given text reflects suicide ideation. "
        "Respond with 'Suicide Watch' if the text suggests the author might be considering suicide. "
        "Respond with 'Depression' if the text indicates the author is experiencing depression but does not suggest an immediate suicide risk.\n\n"
    )
    
    if few_shot_examples:
        for example in few_shot_examples:
            label_description = {
                0: "Depression",
                1: "Suicide Watch"
            }
            prompt += f"Example Text: {example['text']}\nClassification: {label_description[example['label']]}\n\n"
    
    prompt += (
        "Based on the instructions, classify the following text as either 'Depression' or 'Suicide Watch'. "
        "Provide only the classification without additional commentary.\n\n"
        f"Text to Classify: {text}\nClassification:"
    )
    return prompt


def get_prediction(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.InvalidRequestError as e:
        print(f"Encountered an error with the prompt: {e}")
        return "Error: Repetitive pattern detected"

def generate_few_shot_examples(train_data, shots_num):
    # Separate the examples by class
    class_0_examples = [example for example in train_data.to_dict('records') if example['label'] == 0]
    class_1_examples = [example for example in train_data.to_dict('records') if example['label'] == 1]
    
    # Determine the number of examples to select from each class
    num_from_each_class = shots_num // 2
    
    # If shots_num is odd, add one more example to one of the classes randomly
    extra_shot = shots_num % 2
    
    # Select examples
    selected_examples = random.sample(class_0_examples, num_from_each_class) + \
                        random.sample(class_1_examples, num_from_each_class)
    
    # Add the extra example if needed
    if extra_shot:
        extra_class_examples = class_0_examples if random.choice([0, 1]) == 0 else class_1_examples
        selected_examples += random.sample(extra_class_examples, 1)
    
    # Shuffle the selected examples to avoid positional bias
    random.shuffle(selected_examples)
    
    return selected_examples

def eval(json_file_path):
    # Read in the saved JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extract labels and predictions
    true_labels = [result["label"] for result in results]
    predictions = [result["prediction"] for result in results]

    # Map labels and predictions to binary values
    label_map = {"Depression": 0, "Suicide Watch": 1}
    true_binary = [label_map[label] for label in true_labels]
    
    # Track unrecognized predictions
    unrecognized_preds = 0
    
    pred_binary = []
    for pred in predictions:
        if pred in label_map:
            pred_binary.append(label_map[pred])
        else:
            unrecognized_preds += 1
            pred_binary.append(-1)  # Assuming -1 for unrecognized for calculation purposes
    
    # Calculate the percentage of unrecognized predictions
    total_predictions = len(predictions)
    if total_predictions > 0:
        unrecognized_percentage = (unrecognized_preds / total_predictions) * 100
    else:
        unrecognized_percentage = 0
    
    # Calculate accuracy, precision, recall, f1 for recognized predictions only
    accuracy = accuracy_score(true_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='binary', zero_division=0)
    
    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Unrecognized Predictions: {unrecognized_preds} ({unrecognized_percentage:.2f}%)")

    # Append the results to result.txt
    with open("result.txt", "a") as f:
        f.write(f"gpt-3.5-turbo 2-shot\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Unrecognized Predictions: {unrecognized_preds} ({unrecognized_percentage:.2f}%)\n\n")



def main():
    parser = argparse.ArgumentParser(description="Perform few-shot or zero-shot classification with GPT-3.5-turbo")
    parser.add_argument('--shots_num', type=int, default=2, help='Number of shots for few-shot learning')

    args = parser.parse_args()

    train_data, test_data = get_data()
    
    results = []  # To store input, prompt, and predictions
    
    print_example = True
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):        
        few_shot_examples = generate_few_shot_examples(train_data, args.shots_num) if args.shots_num > 0 else None
        prompt = few_shot_prompt(row['text'], few_shot_examples)
        
        if print_example:
            print(prompt)
            print_example = False
        prediction = get_prediction(prompt)
        results.append({
            "input": row['text'],  # Original sentence
            "label": "Depression" if row['label'] == 0 else "Suicide Watch",
            "prompt": prompt,      # Generated prompt
            "prediction": prediction  # Model prediction
        })
    
    json_file_path = f"shot_{args.shots_num}_test_with_predictions.json"
    
    # Save results to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Predictions saved to", json_file_path)
    
    eval(json_file_path)
    
    

if __name__ == "__main__":
    main()
