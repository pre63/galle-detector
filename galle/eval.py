import os

from galle.base import evaluate_text

# Example usage
if __name__ == "__main__":
    sample_to_test_dir = "documents"
    ensemble_parent_dir = "ai_detector_ensemble"
    
    # Get all subdirectories in the ensemble parent dir
    ensemble_dirs = [d for d in os.listdir(ensemble_parent_dir) if os.path.isdir(os.path.join(ensemble_parent_dir, d))]
    ensemble_dirs.sort()  # Sort for consistent order
    
    for filename in os.listdir(sample_to_test_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(sample_to_test_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
            
            for subdir in ensemble_dirs:
                ensemble_dir = os.path.join(ensemble_parent_dir, subdir)
                is_ai, score, total = evaluate_text(text, ensemble_dir=ensemble_dir)
                print(f"{subdir}/File: {filename} | Is AI-generated: {is_ai} (Score: {score}/{total})")