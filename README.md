# QLoRA Fine-tuning of SmolLM-360M on BoolQ Dataset

## Project Overview

This project demonstrates the efficient fine-tuning of Large Language Models (LLMs) using the Quantized Low-Rank Adaptation (QLoRA) technique. QLoRA significantly reduces computational requirements while maintaining high performance. The model used for fine-tuning is `HuggingFaceTB/SmolLM-360M`.

The training dataset utilized is Google's BoolQ dataset, which comprises 15,942 examples of True/False question-answering pairs. Each example includes a question, a passage, and a corresponding answer, with an optional page title for additional context. The primary objective is to enhance the language model's capabilities in Question Answering tasks, which are crucial for applications like Retrieved Augmented Generation (RAG).

## Key Components

- **Model**: `HuggingFaceTB/SmolLM-360M`
- **Fine-tuning Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Dataset**: Google BoolQ (Question-Answering)
- **Libraries**: `datasets`, `peft`, `trl`, `bitsandbytes`, `transformers`, `torch`, `accelerate`

## Experimentation with `r` values

The project explores the impact of different `r` values (rank) in the LoRA configuration on model performance and resource usage. Experiments were conducted with `r` values of 2, 4, 8, and 12, training each configuration for 80 steps.

## Conclusions

1.  **Effect of Rank (`r`)**: The analysis showed that varying the `r` parameter (rank) in the QLoRA configuration resulted in similar convergence behavior in terms of training and validation loss, as well as comparable convergence times.
2.  **Performance Improvement**: The fine-tuned models, across different `r` values, demonstrated a significant improvement in their ability to answer BoolQ questions compared to the non-fine-tuned base model. The base model often generated verbose, irrelevant text, whereas the fine-tuned models provided more concise and task-relevant responses.
3.  **Efficiency of QLoRA**: QLoRA proved to be an effective method for fine-tuning, allowing for adaptation of the LLM to the specific task with reduced computational demands.

This work highlights the efficacy of QLoRA in adapting LLMs for specific tasks like Question Answering, making it a valuable technique for deploying performant models in resource-constrained environments.
