# Egyptian Arabic Post Approval Classifier

A Streamlit web app that reviews Egyptian Arabic social media posts and decides whether a post is approved (موافق) or rejected (مرفوض). It uses a fine-tuned language model from Hugging Face.

## Features

- Accepts Egyptian Arabic text in a text box.
- Sends the post to a fine-tuned causal language model with a moderation prompt.
- Shows the result as approved (✅ البوست مقبول) or rejected (❌ البوست مرفوض).
- Shows the model's raw response for transparency.
- Loads the model once and caches it, so later requests are faster.

## Model

The app loads `tarekkmuhammedd/HyperParameterTuned.NileLLM` from the Hugging Face Hub. It uses GPU (float16) when CUDA is available, and CPU (float32) otherwise.

## Requirements

- Python 3.9 or higher
- Packages: `streamlit`, `transformers`, `torch`
- Internet access on first run, to download the model
- Optional: an NVIDIA GPU with CUDA for faster inference



## Running the App

```bash
streamlit run Posts_Approval_App.py
```

Then open the URL that Streamlit prints, usually http://localhost:8501.

## How It Works

1. The model is loaded once and cached with `@st.cache_resource`.
2. The user enters a post and clicks **تحليل البوست** (Analyze post).
3. The app builds a prompt asking the model to decide whether the post is "موافق" or "مرفوض".
4. The model generates up to 8 new tokens.
5. The app checks the response for approval keywords (`موافق`, `approve`, `accepted`, `تمام`). If it finds one, the post is approved. Otherwise, it is rejected.

## Notes

- The decision depends on keyword matching in the model's output, so unexpected wording may be misclassified.
- `trust_remote_code=True` runs code from the model repository. Only use it with models you trust.
- Sampling (`temperature=0.7`) means the same post may get different results across runs.

## Author

Tarek Muhammad Abdelhamid

