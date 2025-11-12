#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mistral.py
---------------------------------
Run multimodal QA evaluation using LangChain + Ollama models.
Adapted from the original Colab workflow.
"""

import os
import json
import base64
import time
import random
import argparse
import pandas as pd
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# -------------------------------
# Paths and Configurations
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal QA evaluation with Mistral.")
    parser.add_argument("--model", type=str, default="mistral-small3.1:24b",
                        help="Ollama model name (e.g. mistral-small3.1:24b)")
    parser.add_argument("--start", type=int, default=0, help="Start event index")
    parser.add_argument("--end", type=int, default=None, help="End event index (exclusive)")
    parser.add_argument("--dataset", type=str, default="./dataset", help="Path to dataset folder")
    parser.add_argument("--output", type=str, default="./outputs", help="Path to save outputs")
    return parser.parse_args()


# -------------------------------
# Core Functions
# -------------------------------

def write_message(Q_current, input_folder):
    """
    Assemble a LangChain message given the question data and input folder.
    """
    Q_modalities = Q_current["modalities"]
    storm, year, leadtime = Q_current["context"][-3:]

    human_content = []
    for modality in Q_modalities:
        filename = f"{storm}_{year}_{leadtime}h"

        if modality in ["Graphic_Uncertainty_cone", "Graphic_Wind"]:
            image_path = Path(input_folder) / modality / f"{filename}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/gif;base64,{image_b64}"}
                })

        elif modality in ["text_advisory", "Table_wind"]:
            text_path = Path(input_folder) / modality / f"{filename}.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"Text file not found: {text_path}")
            with open(text_path, "r", encoding="utf-8") as text_file:
                advisory_text = text_file.read()
                human_content.append({
                    "type": "text",
                    "text": advisory_text
                })
        else:
            raise ValueError(f"Unknown modality: {modality}")

    # Append question last
    human_content.append({"type": "text", "text": Q_current["question"]})

    messages = [
        SystemMessage(content=Q_current["prompt"]),
        HumanMessage(content=human_content)
    ]

    return messages, Q_current["answer"]


def run_LLM(EventIdx, QASet, EventNames, llm, input_folder, output_folder, model_name):
    """
    Run the LLM for one event index and save output CSV.
    """
    event_name = EventNames[EventIdx]
    print(f"\n>>> Running EventIdx={EventIdx}, EventName={event_name}, LLM={model_name}")

    QASet_thisevent = QASet[event_name]["qa"]
    QASet_thisevent_indexed = list(enumerate(QASet_thisevent))
    random.shuffle(QASet_thisevent_indexed)

    results_dict = {}
    for idx_q, q in QASet_thisevent_indexed:
        t1 = time.time()
        message_thisq, true_answer = write_message(q, input_folder)
        response = llm.invoke(message_thisq)
        t2 = time.time()

        results_dict[idx_q] = {
            "response": response.content.strip(),
            "ground_truth": true_answer,
            "elapsed_sec": round(t2 - t1, 3)
        }

        print(f"Event={EventIdx}, Q_idx={idx_q}, elapsed={t2 - t1:.2f}s")

    # Convert to list sorted by index
    results_sorted = [results_dict[idx] for idx in sorted(results_dict.keys())]

    # Save to CSV
    output_path = Path(output_folder) / f"EvtID{EventIdx}_{event_name}.csv"
    pd.DataFrame(results_sorted).to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")


# -------------------------------
# Main Execution
# -------------------------------
def main():
    args = parse_args()

    dataset_folder = Path(args.dataset)
    input_folder = dataset_folder / "MultiModalInput"
    qa_path = dataset_folder / "CyPortQA.json"
    output_folder = Path(args.output) / args.model.replace(":", "_")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load JSON
    with open(qa_path, "r", encoding="utf-8") as f:
        QASet = json.load(f)
    EventNames = list(QASet.keys())[::-1]

    start_idx = args.start
    end_idx = args.end if args.end is not None else len(EventNames)

    llm = ChatOllama(model=args.model, num_ctx=8192)

    for evnt_idx in range(start_idx, end_idx):
        run_LLM(evnt_idx, QASet, EventNames, llm, input_folder, output_folder, args.model)


if __name__ == "__main__":
    main()