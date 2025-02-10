from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
#model_name = "google/flan-t5-large"
#model_name = "t5-small"
model_name = "distilgpt2"
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=150, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Start a simple chat loop
print("Chatbot: Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chat_with_model(user_input)
    print(f"Chatbot: {response}")
