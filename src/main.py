import pandas as pd
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the dataset
df = pd.read_csv('coursera_course_dataset_v2_no_null.csv')

# Preprocess the data
df['input_text'] = df['Skills'].apply(lambda x: 'Course skills: ' + x)
df['target_text'] = df['Title'].apply(lambda x: 'Course title: ' + x)

training_data = df[['input_text', 'target_text']].values.tolist()

with open('training_data.txt', 'w') as file:
    for input_text, target_text in training_data:
        file.write(input_text + ' \\n ' + target_text + '\n')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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
    num_train_epochs=200,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,  # Log metrics every 100 steps
    disable_tqdm=False,  # Enable tqdm progress bar
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

# Plot the training loss
train_metrics = trainer.state.log_history
print(train_metrics)

train_loss = []
for metric in train_metrics:
    if 'loss' in metric:
        train_loss.append(metric['loss'])

plt.figure(figsize=(12, 6))
plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("training_loss_plot.png")
plt.show()
