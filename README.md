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