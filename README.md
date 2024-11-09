---
title: RAG_-_Project_C_Team_5
app_file: app.py
sdk: gradio
sdk_version: 5.3.0
---
# COS30018 Project C - Team 5 GitHub Repo

**Project C Survival**

Maintained by:
- Alex
- Matt
- Toan

# Installation
Create a Python virtual environment using Python 3.10.4

`git clone` the repository and`cd` into the project repository directory.
Then install the dependencies with:
```bash
pip install -r requirements.txt
```

# Deployment

## Front-end (User Interface)

`cd` to the project repository and activate your Python virtual environment.
You may then run the front-end with Gradio locally using: the command
```bash
python app.py
```

## API

Run the API with the command:
```bash
fastapi dev api.py
```

# Supercomputer Usage (OZstar)

Here is a brief guide for how to use the project on the OZstar supercomputer.

## Setup

1. `cd` to the project folder.
```bash
cd \/fred/oz345/alex
```

2. Load Python 3.10.4.
```bash
module load gcc/11.3.0
module load python/3.10.4
```

3. Activate the virtual environment.
```bash
source team5/bin/activate
```

4. `cd` to the project repository.
```bash
cd \/fred/oz345/alex/ProjectCSurvival
```

5. Set the Hugging Face home directory to the project folder.

NOTE: We're having issues getting this bit to work.
```bash
python
os.environ["HF_HOME"] = "/fred/oz345/alex/cache"
exit()
```

6. Log in to Hugging Face
```bash
huggingface-cli login --token <YOUR HF TOKEN>
```

7. Override GPU (because Toorana nodes don't have GPU so you can't use nvdia-smi)

```bash
CONDA_OVERRIDE_CUDA="11.2" conda install "tensorflow==2.7.0=cuda112*" -c conda-forge
```

## Deployment

To deploy the model on the supercomputer, you will need to submit a batch job using Slurm.

To use GPU to run a job:
- Look into example of ``run_job.sh``
- ``run_job.sh`` use code in ``rag/test.py`` to run
- In command promt, type this to submit job to supercomputer:

```bash
sbatch run_job.sh
```

## Download model to run local (Not working yet)
```bash
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", filename="config.json", cache_dir="/fred/oz345/toan/cache/models/Llama3.1")
```