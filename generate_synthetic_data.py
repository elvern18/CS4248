from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import random
import json 
import re
import pandas as pd 

def is_valid_headline(headline):
    if re.search(r'https?://\S+', headline):
        return False
    if "```" in headline or "def " in headline or "import " in headline or "class " in headline or "[" in headline or "]" in headline:
        return False
    return True


records = []
with open('Sarcasm_Headlines_Dataset.json', 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # Skip empty lines
            record = json.loads(line)
            records.append(record)

data = pd.DataFrame(records)
print(data.head())
sarcastic_data = data[data['is_sarcastic'] == 1]

num_data_to_create = (len(data) - len(sarcastic_data)) - len(sarcastic_data)

corpus = sarcastic_data['headline'].tolist()


# init GPT-2 pipeline
generator = pipeline("text-generation", model="gpt2")


new_rows = []
count = 0 
query = "Generate an original one-liner sarcastic headline inspired by the above context:"
while count < num_data_to_create:

    retrieved_passage = random.choice(corpus)

    prompt = (
        f"Context: {retrieved_passage}\n\n"
        "Write a creative, original, one-line sarcastic headline:"
    )


    result = generator(
        prompt,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,  # Discourage repeating tokens
        num_return_sequences=1,
        truncation=True
    )



    # Get the generated text
    generated_text = result[0]['generated_text']

    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()


    one_liner = generated_text 


    if one_liner in prompt or one_liner.startswith(query) or not is_valid_headline(one_liner):
        continue 

    print(f'context: {retrieved_passage}')
    print("Generated Headline FINAL:", one_liner)
    

    new_row = {"article_link": "", "headline": one_liner, "is_sarcastic": 1}
    new_rows.append(new_row)
    
    count += 1
    print(f"Counter: {count}/{num_data_to_create}\n")


# Create a DataFrame from the new synthetic rows
synthetic_df = pd.DataFrame(new_rows)

# Append the synthetic data to the original DataFrame
data = pd.concat([data, synthetic_df], ignore_index=True)

print("\nUpdated Data (first few rows):")
print(data.head())

data.to_json("Sarcasm_Headlines_Dataset_gpt2.json")