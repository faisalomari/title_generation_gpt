
import pandas as pd

# Load the dataset
df = pd.read_csv('coursera_course_dataset_v2_no_null.csv')

# Preprocess the data
# Assuming 'Skills' are the input and 'Title' is the target output
df['input_text'] = df['Skills'].apply(lambda x: 'Course skills: ' + x)
df['target_text'] = df['Title'].apply(lambda x: 'Course title: ' + x)

# Combine the input and target text
df['training_example'] = df.apply(lambda x: x['input_text'] + ' \\n ' + x['target_text'], axis=1)

# Select the relevant column for training
training_data = df['training_example'].tolist()

# Save the preprocessed data to a new file
with open('training_data.txt', 'w') as file:
    for item in training_data:
        file.write("%s\n" % item)

# Now 'training_data.txt' can be used for fine-tuning the model

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# Make sure to install the transformers library if you haven't already
# pip install transformers

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the dataset
train_path = 'training_data.txt'
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")


from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

# Function to generate course titles
def generate_course_title(skills, model, tokenizer):
    input_text = f"Course skills: {skills}"
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate a sequence of tokens from the input
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated sequence to a readable text
    generated_title = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_title

# Example skills to generate a course title for
skills = "Machine Learning, Data Science, Python Programming"

# Generate the course title
generated_title = generate_course_title(skills, model, tokenizer)
print("Generated Course Title:", generated_title)








# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Load pre-trained model (weights)
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Encode a text input for the model
# text = "Course skills: Machine Learning, Data Science"
# indexed_tokens = tokenizer.encode(text)

# # Convert indexed tokens into a PyTorch tensor
# tokens_tensor = torch.tensor([indexed_tokens])

# # Set the model in evaluation mode to deactivate the DropOut modules
# model.eval()

# # Predict all tokens
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]

# # Get the predicted next sub-word (in our case, the end of the text string)
# predicted_index = torch.argmax(predictions[0, -1, :]).item()
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# # Print the predicted word
# print(predicted_text)

