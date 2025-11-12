#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gemini_flash.py
---------------------------------
Run multimodal QA evaluation using Gemini 2.5 Flash model.
"""

import os,json,base64,time,random,argparse,pandas as pd
from pathlib import Path
import google.generativeai as genai

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--end",type=int,default=None)
    p.add_argument("--dataset",type=str,default="./dataset")
    p.add_argument("--output",type=str,default="./outputs")
    return p.parse_args()

def write_message(Q,input_folder):
    Qm=Q["modalities"];storm,year,lead=Q["context"][-3:];parts=[]
    for m in Qm:
        fn=f"{storm}_{year}_{lead}h"
        if m in ["Graphic_Uncertainty_cone","Graphic_Wind"]:
            img_path=Path(input_folder)/m/f"{fn}.png"
            parts.append(Path(img_path))
        else:
            txt=open(Path(input_folder)/m/f"{fn}.txt","r",encoding="utf-8").read()
            parts.append(txt)
    parts.append(Q["question"])
    return parts,Q["answer"]

def run_LLM(i,QASet,Events,model,input_folder,outdir):
    name=Events[i];print(f"Running {i}: {name}")
    qa=QASet[name]["qa"];random.shuffle(qa);res=[]
    for q in qa:
        parts,gt=write_message(q,input_folder)
        t1=time.time()
        r=model.generate_content(parts)
        t2=time.time()
        res.append({"response":r.text.strip(),"ground_truth":gt,"elapsed_sec":round(t2-t1,3)})
    pd.DataFrame(res).to_csv(Path(outdir)/f"EvtID{i}_{name}.csv",index=False)

def main():
    a=parse_args()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    qa=json.load(open(Path(a.dataset)/"CyPortQA.json","r",encoding="utf-8"))
    Events=list(qa.keys())[::-1]
    model=genai.GenerativeModel("gemini-2.5-flash")
    out=Path(a.output)/"gemini2_5_flash";out.mkdir(parents=True,exist_ok=True)
    for i in range(a.start,a.end or len(Events)):run_LLM(i,qa,Events,model,Path(a.dataset)/"MultiModalInput",out)

if __name__=="__main__":main()
