import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import statistics
import sys

import re

GT_PATH = [f for f in (Path("evaluation") / "ground_truth").iterdir() if f.is_file()]
TEST_PATH = Path("extractor_lib") / "extracted_guidance.jsonl"
# Define the fields to compare (excluding metadata)
COMPARE_FIELDS = [
    "company",
    "guidance_type",
    "metric_name",
    "reporting_period",
    "current_value",
    "unit",
    "guided_range_low",
    "guided_range_high",
    "is_revision"
]

def normalize_string(s):
    if not s:
        return ""
    return str(s).strip().lower()

def normalize_period(p):
    """
    Normalize reporting period.
    e.g. 'FY25' -> 'FY2025'
    'Q1 2025' -> 'Q1 2025'
    """
    s = normalize_string(p)
    # Replace FY25 with FY2025
    s = re.sub(r'fy(\d{2})\b', lambda m: f'fy20{m.group(1)}', s)
    # Replace 2025 Q1 with Q1 2025 (standardize order if needed, but simple replacement helps)
    return s

def normalize_unit(u):
    s = normalize_string(u)
    if s in ['b', 'bn', 'billions']:
        return 'billion'
    if s in ['m', 'mn', 'millions']:
        return 'million'
    return s

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_ground_truth(gt_paths):
    """
    Load ground truth data from multiple files.
    Returns a dictionary mapping uid -> list of guidance items.
    """
    gt_by_uid = {}
    for path in gt_paths:
        print(f"Loading ground truth from {path}...")
        records = load_jsonl(path)
        for record in records:
            uid = record.get('uid')
            if not uid:
                continue
            # Combine gold_standard_guidance and any other guidance fields if present
            guidance_items = record.get('gold_standard_guidance', [])
            if not guidance_items and 'synthetic_guidance' in record: # Handle potential synthetic field name variation
                 guidance_items = record.get('synthetic_guidance', [])
            
            gt_by_uid[uid] = guidance_items
    return gt_by_uid

def load_extracted_data(extracted_path):
    """
    Load extracted data.
    Supports both one guidance item per line (dict) and grouped items per line (list).
    Returns a dictionary mapping uid -> list of guidance items.
    """
    print(f"Loading extracted data from {extracted_path}...")
    extracted_by_uid = defaultdict(list)
    records = load_jsonl(extracted_path)
    for record in records:
        uid = record.get('uid')
        if not uid:
            continue
        
        guidance_data = record.get('guidance')
        if guidance_data:
            if isinstance(guidance_data, list):
                extracted_by_uid[uid].extend(guidance_data)
            else:
                extracted_by_uid[uid].append(guidance_data)
    return extracted_by_uid

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, str(text1), str(text2)).ratio()

def match_items(gt_items, pred_items, threshold=0.6):
    """
    Match predicted items to ground truth items.
    Returns a list of tuples: (gt_item, pred_item)
    Unmatched items will be (gt_item, None) or (None, pred_item).
    """
    matches = []
    unmatched_gt = list(gt_items)
    unmatched_pred = list(pred_items)
    
    # First pass: Try to match by statement_text if available in both
    # Note: GT data might not have statement_text, so we might need a fallback.
    
    # We'll use a greedy approach: find best match, remove, repeat.
    # If statement_text is missing in GT, we'll try to match on (guidance_type, metric_name, reporting_period)
    
    potential_matches = []
    
    for g_idx, gt in enumerate(unmatched_gt):
        for p_idx, pred in enumerate(unmatched_pred):
            score = 0.0
            
            # 1. Try statement_text similarity
            gt_text = gt.get('statement_text') or gt.get('rationales') # Fallback to rationales if statement_text missing
            pred_text = pred.get('statement_text')
            
            text_score = calculate_similarity(gt_text, pred_text)
            
            # 2. Heuristic score based on key fields
            key_field_score = 0.0
            
            # Guidance Type
            if normalize_string(gt.get('guidance_type')) == normalize_string(pred.get('guidance_type')):
                key_field_score += 0.4
            
            # Reporting Period
            if normalize_period(gt.get('reporting_period')) == normalize_period(pred.get('reporting_period')):
                key_field_score += 0.3
            
            # Metric name similarity
            metric_sim = calculate_similarity(gt.get('metric_name'), pred.get('metric_name'))
            key_field_score += 0.3 * metric_sim
            
            # Final score: prefer text match if strong, otherwise key fields
            # If GT lacks text, text_score will be low/zero, so key_field_score dominates.
            if text_score > threshold and gt_text:
                score = text_score
            else:
                score = key_field_score
            
            potential_matches.append({
                'gt_idx': g_idx,
                'pred_idx': p_idx,
                'score': score
            })
    
    # Sort by score descending
    potential_matches.sort(key=lambda x: x['score'], reverse=True)
    
    used_gt_indices = set()
    used_pred_indices = set()
    
    for pm in potential_matches:
        if pm['gt_idx'] in used_gt_indices or pm['pred_idx'] in used_pred_indices:
            continue
        
        # Lower threshold for key field matching since it's a sum of parts
        effective_threshold = threshold
        if not (unmatched_gt[pm['gt_idx']].get('statement_text') or unmatched_gt[pm['gt_idx']].get('rationales')):
             # If we are relying on key fields, we might accept a slightly lower score if it's a strong partial match
             effective_threshold = 0.5

        if pm['score'] >= effective_threshold:
            matches.append((unmatched_gt[pm['gt_idx']], unmatched_pred[pm['pred_idx']]))
            used_gt_indices.add(pm['gt_idx'])
            used_pred_indices.add(pm['pred_idx'])
            
    # Add unmatched
    for i, gt in enumerate(unmatched_gt):
        if i not in used_gt_indices:
            matches.append((gt, None))
            
    for i, pred in enumerate(unmatched_pred):
        if i not in used_pred_indices:
            matches.append((None, pred))
            
    return matches

def score_item(gt, pred):
    """
    Calculate per-field accuracy for a matched pair.
    """
    if not gt:
        # Extra item (False Positive)
        return 0.0, 0 # accuracy, field_count
        
    if not pred:
        # Missing item (False Negative)
        return 0.0, len(COMPARE_FIELDS) # Assuming all fields missed
    
    correct_fields = 0
    total_fields = 0
    
    for field in COMPARE_FIELDS:
        gt_val = gt.get(field)
        
        # Skip null fields in GT? User said: "total_fields = total number of non-null fields"
        if gt_val is None:
            continue
            
        total_fields += 1
        pred_val = pred.get(field)
        
        is_match = False
        
        # Field-specific normalization
        if field == 'reporting_period':
            if normalize_period(gt_val) == normalize_period(pred_val):
                is_match = True
        elif field == 'unit':
            if normalize_unit(gt_val) == normalize_unit(pred_val):
                is_match = True
        elif isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            # 1% tolerance
            if abs(gt_val - pred_val) <= 0.01 * abs(gt_val):
                is_match = True
            elif gt_val == 0 and abs(pred_val) < 1e-6:
                is_match = True
        else:
            # String normalization
            if normalize_string(gt_val) == normalize_string(pred_val):
                is_match = True
                
        if is_match:
            correct_fields += 1
            
    if total_fields == 0:
        return 1.0, 0 # No fields to check
        
    return correct_fields / total_fields, total_fields

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against ground truth.")
    parser.add_argument("--gt-files", nargs='+', required=False, help="Paths to ground truth JSONL files.")
    parser.add_argument("--extracted-file", required=False, help="Path to extracted guidance JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Matching threshold.")
    
    args = parser.parse_args()
    
    # Load data
    if args.gt_files:
        gt_data = load_ground_truth(args.gt_files)
    else:
        gt_data = load_ground_truth(GT_PATH)
    if args.extracted_file:
        extracted_data = load_extracted_data(args.extracted_file)
    else:
        extracted_data = load_extracted_data(TEST_PATH)
    
    # Metrics
    doc_metrics = {"total": 0, "with_guidance_gt": 0, "with_guidance_pred": 0, "correct_detection": 0}
    item_metrics = {"total_gt": 0, "total_pred": 0, "matched": 0, "missing": 0, "extra": 0}
    field_accuracies = []
    
    # Iterate over all UIDs in GT
    all_uids = set(gt_data.keys()) | set(extracted_data.keys())
    
    print(f"\nEvaluating {len(all_uids)} documents...")
    
    for uid in all_uids:
        gt_items = gt_data.get(uid, [])
        pred_items = extracted_data.get(uid, [])
        
        # Document-level metrics
        doc_metrics["total"] += 1
        has_gt = len(gt_items) > 0
        has_pred = len(pred_items) > 0
        
        if has_gt:
            doc_metrics["with_guidance_gt"] += 1
        if has_pred:
            doc_metrics["with_guidance_pred"] += 1
            
        if has_gt and has_pred:
            doc_metrics["correct_detection"] += 1
        elif not has_gt and not has_pred:
            doc_metrics["correct_detection"] += 1
            
        # Item matching
        matches = match_items(gt_items, pred_items, threshold=args.threshold)
        
        for gt, pred in matches:
            if gt and pred:
                item_metrics["matched"] += 1
                acc, count = score_item(gt, pred)
                if count > 0:
                    field_accuracies.append(acc)
            elif gt and not pred:
                item_metrics["missing"] += 1
                # Penalty: 0 score
                # Only add to field accuracies if we want to penalize missing items in the overall field accuracy
                # User said: "If an item is missing -> 0 score for all its fields"
                # So we should add 0.0 to the list?
                # "overall_field_accuracy = mean(all item accuracies)"
                # If we add 0.0, it lowers the mean.
                field_accuracies.append(0.0)
            elif not gt and pred:
                item_metrics["extra"] += 1
                # Penalty for extra item?
                # User said: "If an extra item is extracted -> penalty (false positive)"
                # How to factor into "field accuracy"?
                # Maybe we don't add it to field_accuracies (which measures correctness of *expected* items),
                # but we track it in Precision.
                pass
                
        item_metrics["total_gt"] += len(gt_items)
        item_metrics["total_pred"] += len(pred_items)

    # Calculate aggregates
    precision = item_metrics["matched"] / item_metrics["total_pred"] if item_metrics["total_pred"] > 0 else 0
    recall = item_metrics["matched"] / item_metrics["total_gt"] if item_metrics["total_gt"] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_field_accuracy = statistics.mean(field_accuracies) if field_accuracies else 0.0
    
    print("\n=== Evaluation Results ===")
    print(f"Documents: {doc_metrics['total']}")
    print(f"Document Detection Accuracy: {doc_metrics['correct_detection'] / doc_metrics['total']:.2%}")
    
    print("\n--- Item Level ---")
    print(f"Total GT Items: {item_metrics['total_gt']}")
    print(f"Total Pred Items: {item_metrics['total_pred']}")
    print(f"Matched: {item_metrics['matched']}")
    print(f"Missing (FN): {item_metrics['missing']}")
    print(f"Extra (FP): {item_metrics['extra']}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    print("\n--- Field Level ---")
    print(f"Average Field Accuracy: {avg_field_accuracy:.2%}")
    print("(Calculated over matched items and missing items (as 0))")

if __name__ == "__main__":
    main()
