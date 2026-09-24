# CyPortQA

**CyPortQA** is the first **multimodal benchmark** designed to evaluate how large language models (LLMs and MLLMs) understand and reason about **tropical cyclone impacts on port operations**.  
It connects meteorological forecasts, operational bulletins, and port performance data to test whether models can interpret hazards and recommend realistic operational actions.

---

## 🌊 Overview

Ports are vital nodes in global trade — yet tropical cyclones routinely disrupt maritime logistics and port operations.  
CyPortQA provides a large-scale, structured benchmark to test whether multimodal models can:

1. **Understand** official NOAA/NHC cyclone forecast graphics and advisories  
2. **Estimate** potential disruptions and recovery durations  
3. **Reason** about operational decisions under uncertainty  

Each QA item is grounded in real data from **2015 – 2023**, fusing:
- **NOAA/NHC tropical cyclone forecast products**  
- **USCG port-condition bulletins**  
- **AIS-derived vessel activity and performance data**

The benchmark includes **117 k + QA pairs** organized into three ability categories:

| **Task** | **Core Ability** | **Example Question** |
|-----------|------------------|----------------------|
| **S1 – Situation Understanding** | Spatial reasoning over cyclone maps | “Is Port X inside the uncertainty cone at T − 24 h?” |
| **S2 – Impact Estimation** | Quantitative reasoning over storm impact | “What is the expected recovery duration (hours)?” |
| **S3 – Decision Reasoning** | Operational strategy and policy judgment | “Which port condition should be issued now?” |

---

## 🧭 Repository Structure

```
source_data/
  ├── CyPortQA_Imbalanced/
  ├── NOAA NHC Cyclone Products/
  ├── Port_Condition_Bullitens/
  ├── CyPortQA_template.JSON
  └── Encoded_senario.JSON

dataset/
  ├── CyPortQA.json
  └── MultiModalInput/

models/
  ├── run_mistral.py
  ├── run_gpt4o.py
  ├── run_llama3_2.py
  ├── run_gemini_flash.py
  ├── run_llava1_6.py
  ├── run_gemma3.py
  ├── run_qwen2_5vl.py
  └── run.py          # master runner to execute multiple models

config.json           # select which models and arguments to run
outputs/              # model responses and logs
README.md
```

---

## ⚙️ How to Run

### 1. Set up the environment
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate cyportqa
```

### 2. Choose which models to run
Edit `config.json` to specify your models and arguments:
```json
{
  "models_to_run": ["run_mistral.py", "run_gpt4o.py"],
  "args": {"start": 0, "end": 5, "dataset": "./dataset", "output": "./outputs"}
}
```

### 3. Run benchmark
```bash
python models/run.py
```

Each model script loads the **CyPortQA** dataset, performs multimodal reasoning,  
and saves results automatically to `./outputs/<model_name>/`.

---

## 📈 What’s Inside

- **Dataset:** `CyPortQA.json` — QA pairs across 2,900 + real cyclone–port scenarios  
- **Templates:** `CyPortQA_template.JSON` — 48 question-generation templates  
- **Encoded metadata:** `Encoded_senario.JSON` — storm, port, and lead-time metadata  
- **Source data:** NOAA/NHC cyclone products and USCG bulletins used to construct the benchmark  

Together, these components allow consistent evaluation of multimodal understanding and decision reasoning in extreme-weather contexts.

---

## 📜 License & Citation

Released under the **MIT License**.  
Please cite the following if you use CyPortQA in your research:

```bibtex
@inproceedings{kuai2026cyportqa,
  author    = {Chenchen Kuai and Chenhao Wu and Yang Zhou and Bruce Wang and Tianbao Yang and Zhengzhong Tu and Zihao Li and Yunlong Zhang},
  title     = {{CyPortQA}: Benchmarking Multimodal Large Language Models for Cyclone Preparedness in Port Operation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  volume    = {40},
  number    = {45},
  pages     = {38781--38789},
  doi       = {10.1609/aaai.v40i45.41222},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/41222}
}
```
