import os
import random

import numpy as np
import scipy.stats as stats
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TextClassificationPipeline
from transformers.utils import logging

from galle.base import get_super_maximal_repeats, preprocess_text, train_classifier

logging.set_verbosity(logging.ERROR)

# Main script
if __name__ == "__main__":
  # Load Roy's dataset
  dataset = load_dataset("gsingh1-py/train")
  df = dataset["train"].to_pandas()

  # Filter human and gpt-4o
  subset = None  # small for testing
  human_texts = df["Human_story"].tolist()
  human_texts = [text for text in human_texts if text is not None and text.strip() != ""]
  if subset is not None:
    human_texts = human_texts[:subset]

  gpt_texts = df["GPT_4-o"].tolist()
  gpt_texts = [text for text in gpt_texts if text is not None and text.strip() != ""]
  if subset is not None:
    gpt_texts = gpt_texts[:subset]

  # All documents and labels
  documents = human_texts + gpt_texts

  # Check if any None in the list
  if any(doc is None for doc in documents):
    print(f"Warning: Found None in documents. {documents.count(None)} entries will be skipped.")

  print("Preprocessing texts...")
  documents = [preprocess_text(doc) for doc in documents]
  labels = [0] * len(human_texts) + [1] * len(gpt_texts)
  num_docs = len(documents)

  # Compute super-maximal repeats on the mixed collection
  print("Computing super-maximal repeats...")

  repeats = get_super_maximal_repeats(documents, min_len=20, min_occ=3, mode="char")

  # print 10 sample repeats if available
  if len(repeats) > 9:
    print("Sample super-maximal repeats:")
    for r in repeats[:10]:
      print(f"- {r}")

  # Parameters
  K = 10  # ensemble size
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

  # Known human for semi-supervised
  known_human = [preprocess_text(doc) for doc in human_texts]

  # Create a directory to save models
  save_dir = "ai_detector_ensemble_char"
  os.makedirs(save_dir, exist_ok=True)
  tokenizer.save_pretrained(save_dir)  # Save tokenizer once, as it's shared

  # Ensemble (with safeguards for empty cases)
  print(f"Found {len(repeats)} super-maximal repeats.")
  models = []
  pipe = None  # Will initialize if needed
  while len(models) < K:
    if not repeats:
      break
    subset_size = min(20, len(repeats))
    subset = random.sample(repeats, subset_size)
    pos_texts = [doc for doc in documents if any(r in doc for r in subset)]
    if len(pos_texts) == 0:
      continue
    neg_size = min(len(pos_texts), len(known_human))
    if neg_size == 0:
      continue
    neg_texts = random.sample(known_human, neg_size)
    model = train_classifier(pos_texts, neg_texts, tokenizer)
    models.append(model)
    # Save each model
    model_dir = os.path.join(save_dir, f"model_{len(models)}")
    model.save_pretrained(model_dir)
    print(f"Trained and saved {len(models)} models so far.")

  # Score each doc
  print("Scoring documents...")
  scores = [0] * num_docs
  if models:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m_id, model in enumerate(models):
      if pipe is None or pipe.model != model:
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
      for d_id, doc in enumerate(documents):
        result = pipe(doc[:512])[0]
        if result["label"] == "LABEL_1" and result["score"] > 0.5:
          scores[d_id] += 1
  else:
    print("No classifiers trained (no suitable repeats or positives found). Falling back to zero scores.")

  print("Evaluating performance...")
  # Sort by score descending
  sorted_indices = np.argsort(scores)[::-1]
  # Compute precision at k, e.g., k = len(gpt_texts)
  k = len(gpt_texts)
  top_k_labels = [labels[idx] for idx in sorted_indices[:k]]
  precision = sum(top_k_labels) / k if k > 0 else 0
  print(f"Precision at {k}: {precision}")

  # Compute confusion matrix based on top-k as predicted AI
  total_ai = sum(labels)
  total_human = num_docs - total_ai
  pred_ai_indices = set(sorted_indices[:k])
  tp = sum(1 for idx in range(num_docs) if labels[idx] == 1 and idx in pred_ai_indices)
  fp = len(pred_ai_indices) - tp
  fn = total_ai - tp
  tn = total_human - fp

  # Metrics
  accuracy = (tp + tn) / num_docs if num_docs > 0 else 0
  recall = tp / total_ai if total_ai > 0 else 0
  specificity = tn / total_human if total_human > 0 else 0
  fpr = fp / total_human if total_human > 0 else 0
  fnr = fn / total_ai if total_ai > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  print("\nConfusion Matrix:")
  print(f"TP: {tp}, FP: {fp}")
  print(f"FN: {fn}, TN: {tn}")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall/TPR: {recall:.4f}")
  print(f"Specificity/TNR: {specificity:.4f}")
  print(f"FPR: {fpr:.4f}")
  print(f"FNR: {fnr:.4f}")
  print(f"F1 Score: {f1:.4f}")

  # Statistical tests
  observed_correct = tp + tn
  p_val_binom = stats.binomtest(observed_correct, num_docs, p=0.5, alternative="greater").pvalue if num_docs > 0 else 1.0

  confusion = np.array([[tp, fn], [fp, tn]])
  chi2, p_val_chi2, dof, expected = stats.chi2_contingency(confusion) if min(confusion.shape) > 0 else (0, 1.0, 0, None)

  print("\nStatistical Significance Tests:")
  print(f"Binomial test p-value (accuracy > 0.5): {p_val_binom:.4e}")
  print(f"Chi-square test p-value (independence): {p_val_chi2:.4e}")

  # Conclusion based on thresholds (alpha=0.05)
  alpha = 0.05
  if p_val_binom < alpha and p_val_chi2 < alpha:
    print("\nConclusion: The method shows statistically significant performance better than random guessing ")
    print(f"(binomial p-value = {p_val_binom:.4e} < {alpha}, chi-square p-value = {p_val_chi2:.4e} < {alpha}). ")
    print(
      "This validates the effectiveness of Gallé's method in distinguishing AI-generated (GPT-4o) text from human-written text on the small subset of Roy's dataset."
    )
  else:
    print("\nConclusion: The method does not show statistically significant improvement over chance ")
    print(f"(binomial p-value = {p_val_binom:.4e}, chi-square p-value = {p_val_chi2:.4e}). ")
    print("This is expected with the small test subset and thresholds; for full dataset, use all samples for meaningful validation.")
