# Conversation Summarizer

A transformer-based app that summarizes chat conversations. The model was fine-tuned on the **SAMSum** dialogue dataset using Google Colab, served through a **FastAPI** backend, and used through a **Streamlit** interface.

> **Demo:** <img width="1907" height="872" alt="Screenshot 2026-11-22 004856" src="https://github.com/user-attachments/assets/e07bcfce-4134-4794-99c2-810068cd2bef" />
<img width="1007" height="771" alt="Screenshot 2026-11-22 004943" src="https://github.com/user-attachments/assets/1abacbfe-dc58-42d3-bff3-2b9f6f528d23" />
<img width="1101" height="812" alt="Screenshot 2026-11-22 004956" src="https://github.com/user-attachments/assets/afe52281-54ab-42fb-b3b9-74ca83364d7e" />

## Features

- Summarizes multi-speaker conversations into short, readable summaries
- Fine-tuned transformer model: `[model name, e.g. BART-base / T5-small / Pegasus]`
- Full data preprocessing pipeline on the SAMSum dataset
- REST API built with FastAPI, with request validation using Pydantic
- Simple Streamlit web interface
- Secrets loaded from environment variables (nothing sensitive in the repo)

## Tech Stack

| Area | Tools |
|---|---|
| Model | Hugging Face Transformers, PyTorch |
| Dataset | SAMSum (train / validation / test splits) |
| Training | Google Colab |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Language | Python 3.10+ |

## How It Works

```
Streamlit UI  --->  FastAPI backend  --->  Fine-tuned transformer  --->  Summary
 (conversation)       (/summarize)              (inference)              (output)
```

1. The user pastes a conversation into the Streamlit app.
2. Streamlit sends it to the FastAPI `/summarize` endpoint.
3. Pydantic validates the request.
4. The fine-tuned model generates a summary and returns it as JSON.

## Dataset and Preprocessing

The model is trained on [SAMSum](https://huggingface.co/datasets/samsum), a dataset of chat-style dialogues with human-written summaries, split into training, validation and test sets.

Preprocessing steps:
- `[e.g. removed null / empty entries]`
- `[e.g. cleaned whitespace and special characters]`
- `[e.g. tokenized dialogues and summaries with the model's tokenizer]`
- `[e.g. truncated / padded to max length of X tokens]`

> Replace the bullets above with the steps you actually performed.

## Project Structure

```
text-summarizer/
├── backend/
│   └── main.py              # FastAPI app
├── app.py                   # Streamlit frontend
├── notebooks/
│   ├── preprocessing.ipynb  # Data preprocessing (Jupyter)
│   └── training.ipynb       # Model training (Colab)
├── results/                 # Metrics, plots, sample outputs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> Adjust the file names to match your repo.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then add your values
```

## Model

The fine-tuned weights are too large for GitHub, so they are hosted on Hugging Face:

## Model
The fine-tuned model (T5-small, about 231 MB) is too large for GitHub, so it is hosted on Google Drive.

1. Download the model folder here: [Download my_summarizer_model](https://drive.google.com/drive/folders/1dyWzdcqitRceKhTcto2O7Uns7rmGK1cl?usp=sharing)
2. If Drive gives you a zip file, unzip it.
3. Place the folder named `my_summarizer_model` in the same folder as `app.py`, so the project looks like this:

```
text-summarizer/
├── app.py
├── index.html
├── my_summarizer_model/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── requirements.txt
```

4. Run the app (see the next section).

The backend downloads the model automatically on first run.

## Running the App

Start the backend:

```bash
uvicorn backend.main:app --reload
```

API runs at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

In a second terminal, start the frontend:

```bash
streamlit run app.py
```

## API Example

**POST** `/summarize`

Request:
```json
{
  "text": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you some tomorrow :-)",
  "max_length": 128,
  "min_length": 10
}
```

Response:
```json
{
  "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
}
```

## Training

Training was done on Google Colab (notebook in `notebooks/training.ipynb`).

- **Base model:** `[model name]`
- **Epochs / batch size / learning rate:** `[values]`

## Results

| Metric  | Score |
|---------|-------|
| ROUGE-1 | 42.86 |
| ROUGE-2 | 19.04 |
| ROUGE-L | 34.70 |


## Future Improvements

- Support emails and other long-form text
- Add retrieval (RAG) for summarizing across multiple documents
- Deploy backend and frontend to the cloud
- Add summary length and style controls
  
## Author

**[Daman Dhillon]** | [LinkedIn](https://www.linkedin.com/in/daman-jeet-kaur-dhillon-324191312/) | [GitHub](https://github.com/DamanDhillon16)
