from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")

print(dataset)

# Print the first item in the dataset in a formatted way
print(f"Prompt: {dataset[0]['prompt']}\nChosen: {dataset[0]['chosen']}\nRejected: {dataset[0]['rejected']}")

dataset2 = load_dataset('csv', data_files='convos.csv')['train']

print(dataset2)

print(f"Prompt: {dataset2[0]['prompt']}\nChosen: {dataset2[0]['chosen']}\nRejected: {dataset2[0]['rejected']}")