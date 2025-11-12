#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_llama3_2.py
---------------------------------
Run multimodal QA evaluation using LLaMA 3.2 model via Ollama.
"""
import os,json,base64,time,random,argparse,pandas as pd
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--end",type=int,default=None)
    p.add_argument("--dataset",type=str,default="./dataset")
    p.add_argument("--output",type=str,default="./outputs")
    return p.parse_args()

def write_message(Q,input_folder):
    Qm=Q["modalities"];storm,year,lead=Q["context"][-3:];h=[]
    for m in Qm:
        fn=f"{storm}_{year}_{lead}h"
        if m in ["Graphic_Uncertainty_cone","Graphic_Wind"]:
            path=Path(input_folder)/m/f"{fn}.png"
            b64=base64.b64encode(open(path,"rb").read()).decode("utf-8")
            h.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}})
        else:
            txt=open(Path(input_folder)/m/f"{fn}.txt","r",encoding="utf-8").read()
            h.append({"type":"text","text":txt})
    h.append({"type":"text","text":Q["question"]})
    return [SystemMessage(content=Q["prompt"]),HumanMessage(content=h)],Q["answer"]

def run_LLM(i,QASet,Events,llm,input_folder,outdir):
    name=Events[i];print(f"Running {i}: {name}")
    qa=QASet[name]["qa"];random.shuffle(qa);res=[]
    for q in qa:
        msgs,gt=write_message(q,input_folder)
        t1=time.time();r=llm.invoke(msgs);t2=time.time()
        res.append({"response":r.content.strip(),"ground_truth":gt,"elapsed_sec":round(t2-t1,3)})
    pd.DataFrame(res).to_csv(Path(outdir)/f"EvtID{i}_{name}.csv",index=False)

def main():
    a=parse_args()
    qa=json.load(open(Path(a.dataset)/"CyPortQA.json","r",encoding="utf-8"))
    Events=list(qa.keys())[::-1]
    llm=ChatOllama(model="llama3.2:latest",num_ctx=8192)
    out=Path(a.output)/"llama3_2";out.mkdir(parents=True,exist_ok=True)
    for i in range(a.start,a.end or len(Events)):run_LLM(i,qa,Events,llm,Path(a.dataset)/"MultiModalInput",out)

if __name__=="__main__":main()