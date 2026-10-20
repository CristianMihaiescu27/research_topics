import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


def hop_visualization(values):
    labels = ['one_hop', 'multi_hop']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=['#4C72B0', '#55A868'])

    ax.set_title('Distribution of One-Hop and Multi-Hop Questions')
    ax.set_ylabel('Amount of questions')
    ax.set_xlabel('Category')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval}", ha='center', va='bottom')

    ax.set_ylim(0, 100)
    plt.show()
    
    
def fact_type_visualization(values):
    labels = ['Slow', 'Fast', 'Never']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52'])

    ax.set_title('Speed of facts changing')
    ax.set_ylabel('Amount of questions')
    ax.set_xlabel('Speed')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                f"{yval}", ha='center', va='bottom')

    ax.set_ylim(0, max(values) * 1.15)

    plt.show()


def language_frequency_visualization(values, model):
    labels = ["English", "Chinese", "Vietnamese", "Thai", "Indonesian", "Malay", "Khmer", "Lao", "Dutch", "Romanian"]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    bars = ax.bar(labels, values, color=colors)

    if model == "google/gemma-4-E4B-it":
        ax.set_title('Mean Token Size per Language for google/gemma-4-E4B-it')
    else:
        ax.set_title('Mean Token Size per Language for SeaLLMs-v3-7B-Chat and Qwen/Qwen3-8B')
        

    ax.set_ylabel('Average Tokens per Language')
    ax.set_xlabel('Languages')

    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                f"{yval}", ha='center', va='bottom')

    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()

    plt.show()


def hops_frequency(dataset):
    """
    Distribution of how many hop questions.
    """
    one_hop = 0
    multi_hop = 0
    hops = list(dataset["num_hops"])
    for type in hops:
        if type == "one-hop":
            one_hop += 1
        else:
            multi_hop += 1

    return one_hop, multi_hop


def fact_type_distribution(dataset):
    """
    Compute the distribution of fact types (fast-, slow-, never-changing) in the data.
    """
    slow = 0
    fast = 0
    never = 0
    f_types = list(dataset["fact_type"])
    for f_type in f_types:
        if f_type == "slow-changing":
            slow += 1
        elif f_type == "fast-changing":
            fast += 1
        else:
            never += 1
    
    return fast, slow, never


def count_tokens(questions, tokenizer):
    """
    Returns a dictionary with the amount of tokens in each question.
    """
    return [len(tokenizer(q, add_special_tokens=False)["input_ids"]) for q in questions]


def run_data_stats(dataset):
    MODELS = [
        "Qwen/Qwen3-8B",
        "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "google/gemma-4-E4B-it",
    ]

    tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in MODELS}

    languages = ["English", "Chinese", "Vietnamese", "Thai", "Indonesian", "Malay", "Khmer", "Lao", "Dutch", "Romanian"]

    hop_visualization(hops_frequency(dataset))
    fact_type_visualization(fact_type_distribution(dataset))

    total_nr_questions = len(list(dataset["English"]))


    for model in MODELS:
        word_counts = []
        nr = 0
        for language in languages:
            counts = {name: count_tokens(dataset[language], tok) for name, tok in tokenizers.items()}

            nr = sum(counts[model])

            word_counts.append(nr // total_nr_questions)

        language_frequency_visualization(word_counts, model)


dataset = pd.read_csv("data/questions.csv")

run_data_stats(dataset)

