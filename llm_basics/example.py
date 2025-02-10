
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import load_dataset

# Load a pipeline
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time", max_length=50))

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, how are you?", return_tensors="pt")
print(tokens)

dataset = load_dataset("imdb")
print(dataset["train"][0])

