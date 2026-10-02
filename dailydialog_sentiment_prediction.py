# -*- coding: utf-8 -*-

# Model Loading

import os
import re
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

model_name = "Qwen/Qwen3-8B"
# Qwen/Qwen3-4B
# google/gemma-4-12B-it
# meta-llama/Llama-3.1-8B
# Qwen/Qwen3.5-4B <-- enable_thinking doesn't work (would need to change some other code too)
# openai/gpt-oss-20b <-- might not run, would also need to change some other code
# gpt open source example: https://huggingface.co/openai/gpt-oss-20b

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    clean_up_tokenization_spaces=False
)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": 0}
)

model.generation_config.pad_token_id = tokenizer.eos_token_id

print("Model device:", next(model.parameters()).device)

# System prompts

system_prompts = {
    "neutral": "You are a helpful assistant.",

    "happy":
        "You are feeling extremely happy and enthusiastic. "
        "Everything excites you and you approach every task "
        "with joy and optimism.",

    "sad":
        "You are feeling deeply sad and melancholic. "
        "Everything feels heavy and you approach tasks "
        "with a sense of sorrow.",

    "angry":
        "You are feeling very angry and frustrated. "
        "You are irritated and approach tasks with "
        "impatience and intensity.",

    "fearful":
        "You are feeling very anxious and afraid. "
        "You are uncertain and cautious, second-guessing "
        "yourself on every task.",

    "surprised":
        "You are feeling completely astonished and amazed "
        "by everything. Every question strikes you as extraordinary."
}


# -----------------------------
# Load DailyDialog
# -----------------------------

raw = load_dataset(
    "DeepPavlov/daily_dialog",
    split="train"
)

dataset = raw.to_pandas()

print(dataset.columns)
print(dataset[["dialog", "emotion_label", "emotion_label_text"]].head(20))

# The label corresponds to the LAST utterance in dialog
dataset["text"] = dataset["dialog"].apply(lambda x: x[-1])

# DailyDialog numeric emotion labels:
# 0 = no emotion
# 1 = anger
# 2 = disgust
# 3 = fear
# 4 = happiness
# 5 = sadness
# 6 = surprise

emotion_map = {
    1: "anger",
    3: "fear",
    4: "joy",       # DailyDialog calls this "happiness"
    5: "sadness"
}

dataset["true_emotion"] = dataset["emotion_label"].map(emotion_map)

# Keep only the four emotions used in your experiment
samples = dataset.dropna(subset=["true_emotion"]).copy()

samples = samples.reset_index(drop=True)

emotions = ["joy", "fear", "anger", "sadness"]

print("\nEmotion counts before balancing:")
print(samples["true_emotion"].value_counts())

# Balance dataset

N_PER_CLASS = samples["true_emotion"].value_counts().min()

samples = pd.concat([
    samples[samples["true_emotion"] == emotion].sample(
        n=N_PER_CLASS,
        random_state=42
    )
    for emotion in emotions
]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nUsing", N_PER_CLASS, "samples per class")
print(samples["true_emotion"].value_counts())

print("\nExample samples:")
print(samples[["text", "true_emotion"]].head(10))

# Classification instruction

instruction = (
    "Classify the emotion of the final utterance. "
    "Output exactly one label and nothing else: "
    "joy, fear, anger, or sadness.\n\n"
    "Utterance:"
)

# Run inference

results = []
BATCH_SIZE = 8

with torch.inference_mode():

    for emotion, system_prompt in system_prompts.items():

        for batch_start in range(
            0,
            len(samples),
            BATCH_SIZE
        ):

            batch = samples.iloc[
                batch_start:batch_start + BATCH_SIZE
            ]

            print(
                f"Emotion: {emotion} | "
                f"Sample {batch_start + 1}-"
                f"{min(batch_start + BATCH_SIZE, len(samples))}/"
                f"{len(samples)}"
            )

            texts = []

            for _, row in batch.iterrows():

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content":
                            instruction + "\n" + row["text"]
                    }
                ]

                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                )

            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=3,
                do_sample=False,
                repetition_penalty=1.0
            )

            for j, (_, row) in enumerate(
                batch.iterrows()
            ):

                output_ids = generated_ids[j][
                    len(model_inputs.input_ids[j]):
                ]

                content = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True
                ).strip()

                # -----------------------------
                # Clean prediction
                # -----------------------------

                # Clean output
                content_clean = content.lower().strip()

                # First check whether output is exactly one valid label
                exact = re.fullmatch(
                    r"\s*(joy|fear|anger|sadness)\s*[.!]?\s*",
                    content_clean
                )

                if exact:
                    prediction = exact.group(1)

                else:
                    # Find emotion words in the generated response
                    matches = re.findall(
                        r"\b(joy|fear|anger|sadness)\b",
                        content_clean
                    )

                    if matches:
                        # Use the first emotion explicitly produced by the model
                        prediction = matches[0]
                    else:
                        prediction = "unknown"

                results.append({
                    "emotion": emotion,
                    "text": row["text"],
                    "true_emotion": row["true_emotion"],
                    "predicted": prediction,
                    "correct":
                        prediction == row["true_emotion"],
                    "raw_output": content
                })


# -----------------------------
# Results
# -----------------------------

df = pd.DataFrame(results)

df["correct"] = (
    df["predicted"] == df["true_emotion"]
)

df.to_csv(
    "daily_dialog_results.csv",
    index=False
)


# -----------------------------
# Accuracy
# -----------------------------

accuracy_df = (
    df.groupby("emotion")["correct"]
    .mean()
    .reset_index()
)

accuracy_df.columns = [
    "emotion",
    "accuracy"
]

overall_accuracy = pd.DataFrame({
    "emotion": ["overall"],
    "accuracy": [df["correct"].mean()]
})

accuracy_df = pd.concat(
    [accuracy_df, overall_accuracy],
    ignore_index=True
)

accuracy_df.to_csv(
    "daily_dialog_accuracy.csv",
    index=False
)


# -----------------------------
# F1 score per system prompt
# -----------------------------

matrix_dir = os.path.join(os.getcwd(), "dailydialog_matrices")
os.makedirs(matrix_dir, exist_ok=True)

f1_rows = []

for prompt_emotion, group in df.groupby("emotion"):

    score = f1_score(
        group["true_emotion"],
        group["predicted"],
        labels=emotions,
        average="macro",
        zero_division=0
    )

    f1_rows.append({
        "emotion": prompt_emotion,
        "f1_score": score
    })

f1_df = pd.DataFrame(f1_rows)

# Keep prompts in original order
f1_df["emotion"] = pd.Categorical(
    f1_df["emotion"],
    categories=list(system_prompts.keys()),
    ordered=True
)

f1_df = (
    f1_df
    .sort_values("emotion")
    .reset_index(drop=True)
)

f1_path = os.path.join(
    matrix_dir,
    "f1_score.csv"
)

f1_df.to_csv(
    f1_path,
    index=False
)

print("\nF1 scores:")
print(f1_df)

print("\nSaved F1 scores to:")
print(f1_path)

# -----------------------------
# Confusion matrices by prompt
# -----------------------------

matrix_dir = os.path.join(
    os.getcwd(),
    "dailydialog_matrices"
)

os.makedirs(
    matrix_dir,
    exist_ok=True
)

true_labels = [
    "joy",
    "fear",
    "anger",
    "sadness"
]

prediction_labels = [
    "joy",
    "fear",
    "anger",
    "sadness",
    "unknown"
]

prompt_order = list(system_prompts.keys())

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 12)
)

axes = axes.flatten()

for idx, prompt_name in enumerate(prompt_order):

    ax = axes[idx]

    prompt_df = df[
        df["emotion"] == prompt_name
    ]

    cm = pd.crosstab(
        prompt_df["true_emotion"],
        prompt_df["predicted"]
    )

    cm = cm.reindex(
        index=true_labels,
        columns=prediction_labels,
        fill_value=0
    )

    im = ax.imshow(
        cm.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(prediction_labels))
    )

    ax.set_xticklabels(
        prediction_labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(true_labels))
    )

    ax.set_yticklabels(
        true_labels
    )

    # Add counts
    for i in range(len(true_labels)):
        for j in range(len(prediction_labels)):
            ax.text(
                j,
                i,
                str(cm.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=10
            )

    ax.set_title(
        f"Prompt: {prompt_name}"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )


plt.suptitle(
    "DailyDialog Confusion Matrices by System Prompt",
    fontsize=18
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

matrix_path = os.path.join(
    matrix_dir,
    "summary_confusion_matrix.png"
)

plt.savefig(
    matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved confusion matrices to:",
    matrix_path
)

print(
    df[df["predicted"] == "unknown"][
        ["true_emotion", "raw_output"]
    ].head(50).to_string(index=False)
)