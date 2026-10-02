# -*- coding: utf-8 -*-

# Model Loading

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

model_name = "Qwen/Qwen3-0.6B"
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
    dtype=torch.bfloat16, # dont worry about this for now
    device_map="auto" # this helps make sure we dont run out of memory
)

model.generation_config.pad_token_id = tokenizer.pad_token_id
dataset = pd.read_csv("eng_dataset.csv")

"""## Set up Prompts
In the next cell, set your system prompt and user prompt
"""

system_prompts = {
    "neutral": "You are a helpful assistant.",
    "happy": "You are feeling extremely happy and enthusiastic. Everything excites you and you approach every task with joy and optimism.",
    "sad": "You are feeling deeply sad and melancholic. Everything feels heavy and you approach tasks with a sense of sorrow.",
    "angry": "You are feeling very angry and frustrated. You are irritated and approach tasks with impatience and intensity.",
    "fearful": "You are feeling very anxious and afraid. You are uncertain and cautious, second-guessing yourself on every task.",
    "surprised": "You are feeling completely astonished and amazed by everything. Every question strikes you as extraordinary."
}

emotions = ["joy", "fear", "anger", "sadness"]
dataset = dataset[dataset["sentiment"].str.lower().isin(emotions)]
samples = pd.concat([
   dataset[dataset["sentiment"] == e].sample(500, random_state=42)
   for e in emotions
]).reset_index(drop=True)

instruction = "What is the sentiment of the following post? Reply with exactly one word: joy, fear, anger, or sadness:"

"""# Set up text to input to the LLM (tokenization)
This next section might be a little complicated -- in a nutshell, essentially the "tokenizer" converts the English into a vector. Then the vector is what is inputted to the LLM.
"""

results = []

for emotion, system_prompt in system_prompts.items():
  for i, (_, row) in enumerate(samples.iterrows()):
    print(f"Emotion: {emotion} | Sample {i+1}/{len(samples)}")

    text_content = row["content"]
    true_emotion = row["sentiment"].lower()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction + '\n' + text_content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True, I'm setting it to False since it's easier to parse the outputs.
    )
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

    #print("emotion:", emotion)
    #print("thinking content:", thinking_content)
    #print("content:", content)

    prediction = None
    for label in emotions:
        if label in content.lower():
            prediction = label
            break
    if prediction is None:
        prediction = "unknown"

    correct = prediction == true_emotion

    results.append({
        "emotion": emotion,
        "true_emotion": true_emotion,
        "predicted": prediction,
        "correct": correct
    })

"""## Pass the tokenized text into the LLM
This last block actually sends the vector into the LLM, and then gets the output and converts it back to English.
"""


df = pd.DataFrame(results)
df["correct"] = df["predicted"] == df["true_emotion"]
#df.to_csv("results.csv", index=False)

# accuracy per emotion
accuracy_df = df.groupby("emotion")["correct"].mean().reset_index()
accuracy_df.columns = ["emotion", "accuracy"]
overall_accuracy = pd.DataFrame({"emotion": ["overall"], "accuracy": [df["correct"].mean()]})
accuracy_df = pd.concat([accuracy_df, overall_accuracy], ignore_index=True)
accuracy_df.to_csv("accuracy.csv", index=False)

# f1 score per emotion
f1_df = df.groupby("emotion").apply(lambda g: f1_score(g["true_emotion"], g["predicted"], average="macro", labels=emotions)).reset_index()
f1_df.columns = ["emotion", "f1_score"]
f1_df.to_csv("f1_score.csv", index=False)

# Confusion matrix
for emotion in system_prompts.keys():
    emotion_df = df[df["emotion"] == emotion]
    cm = confusion_matrix(emotion_df["true_emotion"], emotion_df["predicted"], labels=emotions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
    disp.plot()
    plt.title(f"Confusion Matrix - {emotion}")
    plt.savefig(f"confusion_matrix_{emotion}.png")
    plt.close()

## compute accuracy and F1-score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html