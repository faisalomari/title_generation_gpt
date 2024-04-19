from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

# Function to generate course titles
def generate_course_title(skills, model, tokenizer):
    input_text = f"Course skills: {skills}"
    # Encode the input text 
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate a sequence of tokens from the input
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated sequence to a readable text
    generated_title = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text based on the delimiter
    titles = generated_title.split('\n')
    titles = titles[0].split('\\n')
    titles = titles[1].split(':')
    title = titles[1][1:]
    
    return title

# Example skills to generate a course title for
skills = "Linux, Machine Learning, Python Programming"

# Generate the course title
generated_title = generate_course_title(skills, model, tokenizer)
print("Given Course Skills: \\", skills,"\\")
print("Generated Course Title: \\", generated_title,"\\")
