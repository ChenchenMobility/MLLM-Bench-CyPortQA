#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gpt4o.py
---------------------------------
Run multimodal QA evaluation using OpenAI GPT-4o model.
"""

import os, json, base64, time, random, argparse, pandas as pd
from pathlib import Path
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--dataset", type=str, default="./dataset")
    p.add_argument("--output", type=str, default="./outputs")
    return p.parse_args()

def write_message(Q, input_folder):
    Q_modalities = Q["modalities"]
    storm, year, leadtime = Q["context"][-3:]
    human_content = []
    for modality in Q_modalities:
        filename = f"{storm}_{year}_{leadtime}h"
        if modality in ["Graphic_Uncertainty_cone","Graphic_Wind"]:
            path = Path(input_folder)/modality/f"{filename}.png"
            with open(path,"rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
            human_content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}})
        else:
            path = Path(input_folder)/modality/f"{filename}.txt"
            with open(path,"r",encoding="utf-8") as f: text=f.read()
            human_content.append({"type":"text","text":text})
    human_content.append({"type":"text","text":Q["question"]})
    return [{"role":"system","content":Q["prompt"]},
            {"role":"user","content":human_content}], Q["answer"]

def run_LLM(idx, QASet, EventNames, client, input_folder, output_folder):
    name = EventNames[idx]
    print(f"Running Event {idx}: {name}")
    qa = QASet[name]["qa"]
    random.shuffle(qa)
    res=[]
    for q in qa:
        msgs, gt = write_message(q,input_folder)
        t1=time.time()
        r=client.chat.completions.create(model="gpt-4o",messages=msgs)
        t2=time.time()
        res.append({"response":r.choices[0].message.content.strip(),"ground_truth":gt,"elapsed_sec":round(t2-t1,3)})
    pd.DataFrame(res).to_csv(Path(output_folder)/f"EvtID{idx}_{name}.csv",index=False)

def main():
    args=parse_args()
    qa_path=Path(args.dataset)/"CyPortQA.json"
    input_folder=Path(args.dataset)/"MultiModalInput"
    output_folder=Path(args.output)/"gpt4o"
    output_folder.mkdir(parents=True,exist_ok=True)
    QASet=json.load(open(qa_path,"r",encoding="utf-8"))
    Events=list(QASet.keys())[::-1]
    client=OpenAI()
    for i in range(args.start, args.end or len(Events)):
        run_LLM(i,QASet,Events,client,input_folder,output_folder)

if __name__=="__main__": main()