import os
import random
import re
from bisect import bisect_left

import markdown
import numpy as np
import scipy.stats as stats
import supermaxrep
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers.utils import logging

logging.set_verbosity(logging.ERROR)


# Function to get super-maximal repeats using the library (word-based)
def get_super_maximal_repeats(documents, min_len=10, min_occ=3, mode="word"):  # word or char
  if not documents:
    return []
  repeat_objs = supermaxrep.find_supermaximal_repeats_docs(documents, min_len=min_len, min_occ=min_occ, mode=mode)
  repeats = [r.text for r in repeat_objs]
  return list(set(repeats))  # Ensure unique, though library provides unique SMRs


# Custom dataset for trainer
class TextDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_length=512):
    self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


# Function to train a classifier using manual PyTorch loop (no Trainer/accelerate needed)
def train_classifier(pos_texts, neg_texts, tokenizer):
  texts = pos_texts + neg_texts
  labels = [1] * len(pos_texts) + [0] * len(neg_texts)
  dataset = TextDataset(texts, labels, tokenizer)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
  model.to(device)

  dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
  optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

  num_epochs = 3
  model.train()
  for epoch in range(num_epochs):
    for batch in dataloader:
      optimizer.zero_grad()
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels_batch = batch["labels"].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()

  return model


def preprocess_text(text):
  if text is None:
    text = ""  # Handle None texts

  # Remove common sentences from AI generated text
  exact_matches = [
    "[Insert Date]",
    "By [Your Name]",
    "[Your Name]",
    "[Your City]",
    "[Insert City]",
    "[Insert Date]",
    "Follow us on Twitter NYTimes",
    "Follow us on Facebook NYTimes",
    "Follow us on Instagram NYTimes",
  ]
  regex_patterns = [r"\[Your \([a-zA-Z ]+\)\]", r"\[Insert [a-zA-Z ']+\]"]

  for sentence in exact_matches:
    text = re.sub(re.escape(sentence), " ", text, flags=re.IGNORECASE)

  for pattern in regex_patterns:
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

  # Convert Markdown to HTML and extract plain text with BeautifulSoup
  html = markdown.markdown(text)
  soup = BeautifulSoup(html, "html.parser")
  text = soup.get_text()

  # Additional cleanups: remove URLs, emails, etc.
  text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)  # Remove URLs
  text = re.sub(r"\S+@\S+", " ", text)  # Remove emails
  text = re.sub(r"[^a-zA-Z\s.,!?;]", " ", text)  # Keep letters, spaces, basic punctuation

  # Collapse multiple newlines and spaces while preserving line breaks
  text = re.sub(r"\n+", "\n", text)
  text = re.sub(r" +", " ", text)
  text = text.strip()

  # remove all words with less than 3 characters
  text = " ".join([word for word in text.split() if len(word) >= 3])

  return text


# Evaluation function
def evaluate_text(text, ensemble_dir="ai_detector_ensemble", threshold=0.5, score_threshold=5):
  """
    Evaluate if a text is likely AI-generated.

    Args:
    - text: The input text to evaluate (str)
    - ensemble_dir: Directory where models are saved (str)
    - threshold: Confidence threshold for individual model predictions (float)
    - score_threshold: Minimum ensemble score to classify as AI-generated (int, e.g., > half of models)

    Returns:
    - is_ai: True if likely AI-generated, False otherwise (bool)
    - score: Number of models voting for AI-generated (int)
    - total_models: Total models in ensemble (int)
    """
  processed_text = preprocess_text(text)
  if not processed_text:
    return False, 0, 0  # Empty text, can't classify

  tokenizer = AutoTokenizer.from_pretrained(ensemble_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  models = []
  for subdir in sorted(os.listdir(ensemble_dir)):
    model_path = os.path.join(ensemble_dir, subdir)
    if os.path.isdir(model_path) and subdir.startswith("model_"):
      model = AutoModelForSequenceClassification.from_pretrained(model_path)
      model.to(device)
      models.append(model)

  if not models:
    raise ValueError("No models found in the ensemble directory.")

  score = 0
  pipe = None
  for model in models:
    if pipe is None or pipe.model != model:
      pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    result = pipe(processed_text[:512])[0]  # Truncate to max length
    if result["label"] == "LABEL_1" and result["score"] > threshold:
      score += 1

  is_ai = score > score_threshold
  return is_ai, score, len(models)
