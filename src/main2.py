import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('coursera_course_dataset_v2_no_null.csv')

# Preprocessing the data
df['input_text'] = df['Skills'].apply(lambda x: 'Course skills: ' + x)
df['target_text'] = df['Title'].apply(lambda x: 'Course title: ' + x)

# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the preprocessed data to new files
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# Tokenizer and Model Initialization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training Dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='train_data.csv',
    block_size=128
)

# Validation Dataset
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='val_data.csv',
    block_size=128
)

# Test Dataset
test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='test_data.csv',
    block_size=128
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=70,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",  # Evaluate every `logging_steps`
    eval_steps=500,  # Perform evaluation every 500 steps
    disable_tqdm=False,
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Initialize empty lists to store validation loss and accuracy
val_loss_list = []
val_accuracy_list = []

# Define function to compute accuracy
def compute_accuracy(pred):
    labels_ids = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels_ids).mean()}

# Define callback function to log validation loss and accuracy
def log_validation_metrics_callback(trainer, eval_dataloader):
    eval_result = trainer.evaluate(eval_dataloader=eval_dataloader)
    val_loss_list.append(eval_result['eval_loss'])
    val_accuracy_list.append(eval_result['accuracy'])
    print(f"Validation Loss: {eval_result['eval_loss']}, Accuracy: {eval_result['accuracy']}")

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# Evaluate on test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)

# Print accuracy of the validation and test sets
print("Validation Accuracy:", val_accuracy_list[-1])
print("Test Accuracy:", test_results['eval_accuracy'])

# Plot the training and validation loss
train_metrics = trainer.state.log_history
train_loss = [metric['loss'] for metric in train_metrics if 'loss' in metric]

plt.figure(figsize=(12, 6))
plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
plt.plot(range(len(val_loss_list)), val_loss_list, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("training_validation_loss_plot.png")
plt.show()
