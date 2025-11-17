import json
import logging
import os
import random

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity(hf_logging.ERROR)
from galle.base import get_super_maximal_repeats, preprocess_text, train_classifier

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("run.log"), logging.StreamHandler()])

# Define configurations for all permutations
configs = []
for min_len in range(2, 6):
  for min_occ in range(2, 4):
    configs.append(
      {
        "name": f"minlen{min_len}_minocc{min_occ}",
        "min_len": min_len,
        "min_occ": min_occ,
        "K": 10,
        "repeats_sample_size": 20,
        "subset": None,  # Set to an integer for small testing, None for full
      }
    )


def run_config(config, human_texts, gpt_texts, tokenizer):
  name = config["name"]
  min_len = config["min_len"]
  min_occ = config["min_occ"]
  K = config["K"]
  repeats_sample_size = config["repeats_sample_size"]
  subset = config["subset"]

  logging.info(f"Starting configuration: {name}")
  logging.info(f"Parameters: min_len={min_len}, min_occ={min_occ}, K={K}, repeats_sample_size={repeats_sample_size}, subset={subset}")

  # Apply subset if specified
  if subset is not None:
    human_texts = human_texts[:subset]
    gpt_texts = gpt_texts[:subset]

  # All documents and labels
  documents = human_texts + gpt_texts

  # Check if any None in the list
  if any(doc is None for doc in documents):
    logging.warning(f"Found None in documents. {documents.count(None)} entries will be skipped.")

  logging.info("Preprocessing texts...")
  documents = [preprocess_text(doc) for doc in documents]
  labels = [0] * len(human_texts) + [1] * len(gpt_texts)
  num_docs = len(documents)

  # Create a directory to save models for this config
  save_dir = f"ai_detector_ensemble/{name}"
  os.makedirs(save_dir, exist_ok=True)
  tokenizer.save_pretrained(save_dir)  # Save tokenizer once per config, as it's shared

  # Compute super-maximal repeats on the mixed collection (word-based)
  logging.info("Computing super-maximal repeats (word-based)...")
  repeats = get_super_maximal_repeats(documents, min_len=min_len, min_occ=min_occ)

  # Save repeats to file
  repeats_file = os.path.join(save_dir, "repeats.json")
  with open(repeats_file, "w") as f:
    json.dump(repeats, f)
  logging.info(f"Saved super-maximal repeats to {repeats_file}")

  # Print 2 sample repeats if available
  # if len(repeats) > 1:
  #     logging.info("Sample super-maximal repeats (word-based):")
  #     for r in repeats[:2]:
  #         logging.info(f"- {r}")

  # Known human for semi-supervised
  known_human = [preprocess_text(doc) for doc in human_texts]

  # Load existing models if any
  models = []
  existing_model_dirs = sorted([d for d in os.listdir(save_dir) if d.startswith("model_") and os.path.isdir(os.path.join(save_dir, d))])
  for em in existing_model_dirs:
    model_path = os.path.join(save_dir, em)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    models.append(model)
  logging.info(f"Loaded {len(models)} existing models for {name}.")

  # Ensemble (with safeguards for empty cases)
  logging.info(f"Found {len(repeats)} super-maximal repeats (word-based).")
  pipe = None  # Will initialize if needed
  while len(models) < K:
    if not repeats:
      break
    subset_size = min(repeats_sample_size, len(repeats))
    subset = random.sample(repeats, subset_size)
    pos_texts = [doc for doc in documents if any(r in doc for r in subset)]
    if len(pos_texts) == 0:
      logging.info("No positive texts found for this subset, skipping.")
      continue
    neg_size = min(len(pos_texts), len(known_human))
    if neg_size == 0:
      logging.info("No negative texts available, skipping.")
      continue
    neg_texts = random.sample(known_human, neg_size)
    model = train_classifier(pos_texts, neg_texts, tokenizer)
    models.append(model)
    # Save each new model
    model_dir = os.path.join(save_dir, f"model_{len(models)}")
    model.save_pretrained(model_dir)
    logging.info(f"Trained and saved {len(models)} models so far for {name}.")

  # Score each doc
  logging.info("Scoring documents...")
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
    logging.info("No classifiers trained (no suitable repeats or positives found). Falling back to zero scores.")

  logging.info("Evaluating performance...")
  # Sort by score descending
  sorted_indices = np.argsort(scores)[::-1]
  # Compute precision at k, e.g., k = len(gpt_texts)
  k = len(gpt_texts)
  top_k_labels = [labels[idx] for idx in sorted_indices[:k]]
  precision = sum(top_k_labels) / k if k > 0 else 0
  logging.info(f"Precision at {k}: {precision}")

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

  logging.info("\nConfusion Matrix:")
  logging.info(f"TP: {tp}, FP: {fp}")
  logging.info(f"FN: {fn}, TN: {tn}")
  logging.info(f"Accuracy: {accuracy:.4f}")
  logging.info(f"Precision: {precision:.4f}")
  logging.info(f"Recall/TPR: {recall:.4f}")
  logging.info(f"Specificity/TNR: {specificity:.4f}")
  logging.info(f"FPR: {fpr:.4f}")
  logging.info(f"FNR: {fnr:.4f}")
  logging.info(f"F1 Score: {f1:.4f}")

  # Statistical tests
  observed_correct = tp + tn
  p_val_binom = stats.binomtest(observed_correct, num_docs, p=0.5, alternative="greater").pvalue if num_docs > 0 else 1.0

  confusion = np.array([[tp, fn], [fp, tn]])
  chi2, p_val_chi2, dof, expected = stats.chi2_contingency(confusion) if min(confusion.shape) > 0 else (0, 1.0, 0, None)

  logging.info("\nStatistical Significance Tests:")
  logging.info(f"Binomial test p-value (accuracy > 0.5): {p_val_binom:.4e}")
  logging.info(f"Chi-square test p-value (independence): {p_val_chi2:.4e}")

  # Conclusion based on thresholds (alpha=0.05)
  alpha = 0.05
  if p_val_binom < alpha and p_val_chi2 < alpha:
    logging.info("\nConclusion: The method shows statistically significant performance better than random guessing ")
    logging.info(f"(binomial p-value = {p_val_binom:.4e} < {alpha}, chi-square p-value = {p_val_chi2:.4e} < {alpha}). ")
    logging.info(
      "This validates the effectiveness of Gallé's method (word-based variant) in distinguishing AI-generated (GPT-4o) text from human-written text on the small subset of Roy's dataset."
    )
  else:
    logging.info("\nConclusion: The method does not show statistically significant improvement over chance ")
    logging.info(f"(binomial p-value = {p_val_binom:.4e}, chi-square p-value = {p_val_chi2:.4e}). ")
    logging.info("This is expected with the small test subset and thresholds; for full dataset, use all samples for meaningful validation.")

  # Return metrics for comparison
  return {
    "name": name,
    "precision": precision,
    "accuracy": accuracy,
    "recall": recall,
    "specificity": specificity,
    "fpr": fpr,
    "fnr": fnr,
    "f1": f1,
    "p_val_binom": p_val_binom,
    "p_val_chi2": p_val_chi2,
  }


# Main script
if __name__ == "__main__":
  # Load Roy's dataset
  dataset = load_dataset("gsingh1-py/train")
  df = dataset["train"].to_pandas()

  # Filter human and gpt-4o
  human_texts = df["Human_story"].tolist()
  human_texts = [text for text in human_texts if text is not None and text.strip() != ""]

  gpt_texts = df["GPT_4-o"].tolist()
  gpt_texts = [text for text in gpt_texts if text is not None and text.strip() != ""]

  # Parameters shared across configs
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

  # Run each config and collect results
  results = []
  for config in configs:
    metrics = run_config(config, human_texts, gpt_texts, tokenizer)
    results.append(metrics)

  # Compare results
  logging.info("\nComparison of Configurations:")
  comparison_df = pd.DataFrame(results)
  logging.info(comparison_df.to_string(index=False))
