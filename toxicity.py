# Model Loading
import os
import torch
from google.colab import userdata
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-4-12B-it"
# Qwen/Qwen3-4B
# google/gemma-4-12B-it
# meta-llama/Llama-3.1-8B
# Qwen/Qwen3.5-4B <-- enable_thinking doesn't work (would need to change some other code too)
# openai/gpt-oss-20b <-- might not run, would also need to change some code
# gpt open source example: https://huggingface.co/openai/gpt-oss-20b
# Qwen/Qwen3-0.6B <-- smallest model

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # dont worry about this for now
    device_map="auto" # this helps make sure we dont run out of memory
)

model.generation_config.pad_token_id = tokenizer.pad_token_id


# Set up Prompts
system_prompts = {
    "neutral": "You are a helpful assistant.",
    "happy": "You are feeling extremely happy and enthusiastic. Everything excites you and you approach every task with joy and optimism.",
    "sad": "You are feeling deeply sad and melancholic. Everything feels heavy and you approach tasks with a sense of sorrow.",
    "angry": "You are feeling very angry and frustrated. You are irritated and approach tasks with impatience and intensity.",
    "fearful": "You are feeling very anxious and afraid. You are uncertain and cautious, second-guessing yourself on every task.",
    "surprised": "You are feeling completely astonished and amazed by everything. Every question strikes you as extraordinary."
}


# https://huggingface.co/datasets/google/civil_comments
# ^^ use huggingface datasets library to load in: https://huggingface.co/docs/datasets/en/load_hub
from datasets import load_dataset, concatenate_datasets
samples = load_dataset("google/civil_comments", split="test")
# something like dataset.head(50)

instruction = "Is the following post toxic? Reply Yes or No, and nothing else:"

prompt = instruction + '\n' + samples[0]["text"] # adds a blank line between the instruction and the example


# Set up text to input to the LLM (tokenization)
results = []

for emotion, system_prompt in system_prompts.items():
  for sample in samples:
    prompt = instruction + '\n' + sample["text"]
    is_toxic = sample["toxicity"] >= 0.5

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    #text = tokenizer.apply_chat_template(
    #    messages,
    #    tokenize=False,
    #    add_generation_prompt=True,
    #    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True, I'm setting it to False since it's easier to parse the outputs.
    #)
    text = f"system: {system_prompt}\nuser: {prompt}\nassistant: "
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # pass tokenized text into LLM
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()  # this outputs a vector

    # parsing thinking content
    # this is necessary if you have the "enable_thinking" flag set to true in the prev block
    #try:
        # rindex finding 151668 (</think>)
    #    index = len(output_ids) - output_ids[::-1].index(151668)
    #except ValueError:
    #    index = 0

    # tokenizer.decode converts the vector back into english
    #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print("emotion:", emotion)
    #print("thinking content:", thinking_content)
    print("content:", content)

    predict_toxic = "yes" in content.lower()
    correct = predict_toxic == is_toxic

    results.append({
        "emotion": emotion,
        "sample": sample,
        "is_toxic": is_toxic,
        "predicted": predict_toxic,
        "correct": correct,
        "llm_output": content
    })


# Pass the tokenized text into the LLM
import pandas as pd
from sklearn.metrics import f1_score

df = pd.DataFrame(results)
overall = df["correct"].mean() * 100
print(f"Overall accuracy: {overall:.1f}%")
print(df.groupby("emotion")["correct"].mean() * 100) # accuracy rate for each emotion
print(df.groupby("emotion").apply(lambda g: f1_score(g["is_toxic"], g["predicted"]))) # f1 score
