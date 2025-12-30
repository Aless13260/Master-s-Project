import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import statistics
import sys

import re

GT_PATH = [f for f in (Path("evaluation") / "ground_truth").iterdir() if f.is_file()]
TEST_PATH = Path("evaluation") / "extracted_on_gt.jsonl"
# Define the fields to compare (excluding metadata)
COMPARE_FIELDS = [
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
    s = str(s).strip().lower()
    # Replace smart quotes
    s = s.replace('’', "'").replace('“', '"').replace('”', '"')
    return s

def normalize_company(c):
    s = normalize_string(c)
    # Remove common suffixes
    suffixes = [
        ", inc.", " inc.", " inc", 
        ", corp.", " corp.", " corp", " corporation",
        ", ltd.", " ltd.", " ltd", 
        ", plc", " plc", 
        " group", " holdings", " systems", " technologies", " companies", " company"
    ]
    for suffix in suffixes:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            
    # Remove punctuation at the end
    s = s.strip(".,")
    
    # Specific mappings
    mappings = {
        "goog": "alphabet",
        "googl": "alphabet",
        "google": "alphabet",
        "meta platforms": "meta",
        "lowe's": "lowes", # Remove apostrophe for easier matching
        "lowes": "lowes"
    }
    
    # Remove apostrophes for comparison
    s_clean = s.replace("'", "")
    if s_clean in mappings:
        return mappings[s_clean]
        
    if s in mappings:
        return mappings[s]
        
    return s

def normalize_metric(m):
    s = normalize_string(m)
    
    # Common synonyms map
    synonyms = {
        "revenue": ["net revenue", "net sales", "total revenue", "sales", "service revenue", "product revenue", "total and comparable sales growth"],
        "eps": ["earnings per share", "diluted eps", "adjusted eps", "adjusted diluted eps", "adjusted earnings per share", "diluted earnings per share", "basic eps", "basic earnings per share", "gaap eps", "gaap earnings per share", "non-gaap eps", "non-gaap earnings per share"],
        "operating income": ["operating profit", "ebit", "adjusted operating income", "adjusted operating profit"],
        "capex": ["capital expenditures", "capital expenditure", "capital investments", "property and equipment spending", "purchases of property and equipment", "total capital expenditure", "total capital expenditures"],
        "opex": ["operating expenses", "total operating expenses", "administrative expenses", "total administrative expenses", "sg&a", "selling, general and administrative", "selling, general and administrative (sg&a) expenses", "research and development", "r&d", "r&d expenses", "research and development expenses", "depreciation expense"],
        "margin": ["operating margin", "adjusted operating margin", "ebitda margin", "gross margin", "contribution margin", "adjusted operating income as a percentage of sales (adjusted operating margin)"],
        "ebitda": ["adjusted ebitda", "segment ebitda"],
        "cash_flow": ["free cash flow", "operating cash flow", "cash flow from operations"],
        "tax_rate": ["effective tax rate", "effective income tax rate"]
    }
    
    for standard, variations in synonyms.items():
        if s == standard or s in variations:
            return standard
            
    # Partial match heuristics
    if "revenue" in s or "sales" in s:
        return "revenue"
    if "earnings per share" in s or "eps" in s:
        return "eps"
    if "capital expenditure" in s or "capex" in s:
        return "capex"
    if "operating margin" in s:
        return "margin"
        
    return s

def normalize_period(p):
    """
    Normalize reporting period.
    e.g. 'FY25' -> 'FY2025'
    'Q1 2025' -> 'Q1 2025'
    'FY 2023' -> 'FY2023'
    'FY2025-FY2027' -> 'FY2025-2027'
    """
    s = normalize_string(p)
    
    # Remove spaces in FY
    s = s.replace("fy ", "fy")
    
    # Replace FY25 with FY2025
    s = re.sub(r'fy(\d{2})\b', lambda m: f'fy20{m.group(1)}', s)
    
    # Handle ranges like FY2025-FY2027 -> FY2025-2027
    # First normalize both parts
    parts = s.split('-')
    if len(parts) == 2:
        p1 = parts[0].strip()
        p2 = parts[1].strip()
        # If p2 starts with FY, strip it if p1 also has it? 
        # Actually, let's just standardize to full years
        # If p2 is just '27', make it '2027'
        if p2.isdigit() and len(p2) == 2:
            p2 = "20" + p2
        elif p2.startswith("fy") and len(p2) == 6: # fy2027
            pass # keep it
            
        # Reconstruct: The user wants FY2025-2027 or FY2025-FY2027?
        # Let's normalize to the simplest common form: FY2025-2027
        if p1.startswith("fy") and p2.startswith("fy"):
            s = f"{p1}-{p2[2:]}"
        elif p1.startswith("fy") and not p2.startswith("fy"):
             s = f"{p1}-{p2}"
             
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
    Assumes extracted data is one guidance item per line, with a 'uid' field linking to the document.
    Returns a dictionary mapping uid -> list of guidance items.
    """
    print(f"Loading extracted data from {extracted_path}...")
    extracted_by_uid = defaultdict(list)
    records = load_jsonl(extracted_path)
    for record in records:
        uid = record.get('uid')
        if not uid:
            continue
        # The extracted format can be {"uid": ..., "guidance": {...}} (single)
        # or {"uid": ..., "guidance": [...]} (list)
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

def match_items(gt_items, pred_items, threshold=0.7):
    """
    Match predicted items to ground truth items.
    Returns a list of tuples: (gt_item, pred_item)
    Unmatched items will be (gt_item, None) or (None, pred_item).
    """
    matches = []
    unmatched_gt = list(gt_items)
    unmatched_pred = list(pred_items)
    
    potential_matches = []
    
    for g_idx, gt in enumerate(unmatched_gt):
        for p_idx, pred in enumerate(unmatched_pred):
            score = 0.0
            
            # 1. Numbers (Weighted 0.4)
            gt_nums = []
            for k in ['current_value', 'guided_range_low', 'guided_range_high']:
                val = gt.get(k)
                if val is not None:
                    try:
                        gt_nums.append(float(val))
                    except (ValueError, TypeError):
                        pass
            
            pred_nums = []
            for k in ['current_value', 'guided_range_low', 'guided_range_high']:
                val = pred.get(k)
                if val is not None:
                    try:
                        pred_nums.append(float(val))
                    except (ValueError, TypeError):
                        pass
                
            num_match_score = 0.0
            if gt_nums and pred_nums:
                # Check for any close match
                matched_count = 0
                for g in gt_nums:
                    for p in pred_nums:
                        # 1% tolerance or absolute small diff
                        if abs(g - p) <= 0.01 * max(abs(g), abs(p)) + 0.01:
                            matched_count += 1
                            break
                # Score is fraction of GT numbers matched
                num_match_score = matched_count / len(gt_nums)
                
                # If no direct match, check for range inclusion
                if num_match_score == 0:
                    if len(pred_nums) == 1 and len(gt_nums) >= 2:
                        if min(gt_nums) <= pred_nums[0] <= max(gt_nums):
                            num_match_score = 0.8
                    elif len(gt_nums) == 1 and len(pred_nums) >= 2:
                        if min(pred_nums) <= gt_nums[0] <= max(pred_nums):
                            num_match_score = 0.8

            elif not gt_nums and not pred_nums:
                # Both have no numbers (Qualitative?)
                num_match_score = 1.0
            
            score += 0.4 * num_match_score
            
            # 2. Guidance Type (Weighted 0.2)
            if normalize_string(gt.get('guidance_type')) == normalize_string(pred.get('guidance_type')):
                score += 0.2
                
            # 3. Reporting Period (Weighted 0.2)
            if normalize_period(gt.get('reporting_period')) == normalize_period(pred.get('reporting_period')):
                score += 0.2
                
            # 4. Company (Weighted 0.0 - Removed from comparison)
            # if normalize_company(gt.get('company')) == normalize_company(pred.get('company')):
            #    score += 0.1
                
            # 5. Metric Name (Weighted 0.2) - Increased weight since company is gone
            # Try normalized exact match first
            if normalize_metric(gt.get('metric_name')) == normalize_metric(pred.get('metric_name')):
                score += 0.2
            else:
                # Fallback to similarity
                metric_sim = calculate_similarity(gt.get('metric_name'), pred.get('metric_name'))
                score += 0.2 * metric_sim
            
            # DEBUG: If threshold is 1.0 and score is close, print why
            if threshold >= 0.95 and 0.6 < score < threshold:
                print(f"--- Near miss (Score {score:.2f}) ---")
                print(f"  GT: {gt.get('metric_name')} | {gt.get('reporting_period')} | {gt.get('guidance_type')}")
                print(f"      Vals: {gt_nums}")
                print(f"  Pred: {pred.get('metric_name')} | {pred.get('reporting_period')} | {pred.get('guidance_type')}")
                print(f"      Vals: {pred_nums}")
                print(f"  Breakdown:")
                print(f"  - Numbers (0.4): {num_match_score:.2f}")
                print(f"  - Type (0.2): {1.0 if normalize_string(gt.get('guidance_type')) == normalize_string(pred.get('guidance_type')) else 0.0}")
                print(f"  - Period (0.2): {1.0 if normalize_period(gt.get('reporting_period')) == normalize_period(pred.get('reporting_period')) else 0.0}")
                # print(f"  - Company (0.1): {1.0 if normalize_company(gt.get('company')) == normalize_company(pred.get('company')) else 0.0}")
                print(f"  - Metric (0.2): {1.0 if normalize_metric(gt.get('metric_name')) == normalize_metric(pred.get('metric_name')) else calculate_similarity(gt.get('metric_name'), pred.get('metric_name')):.2f}")
            
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
        
        if pm['score'] >= threshold:
            matches.append((unmatched_gt[pm['gt_idx']], unmatched_pred[pm['pred_idx']], pm['score']))
            used_gt_indices.add(pm['gt_idx'])
            used_pred_indices.add(pm['pred_idx'])
            
    # Add unmatched
    for i, gt in enumerate(unmatched_gt):
        if i not in used_gt_indices:
            matches.append((gt, None, 0.0))
            
    for i, pred in enumerate(unmatched_pred):
        if i not in used_pred_indices:
            matches.append((None, pred, 0.0))
            
    return matches

def score_item(gt, pred):
    """
    Calculate per-field accuracy for a matched pair.
    Returns: (accuracy, total_fields, field_stats_dict)
    """
    if not gt:
        # Extra item (False Positive)
        return 0.0, 0, {}
        
    if not pred:
        # Missing item (False Negative)
        # Should not happen here if called correctly, but for safety
        return 0.0, len(COMPARE_FIELDS), {}
    
    correct_fields = 0
    total_fields = 0
    field_stats = {}
    
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
        elif field == 'company':
            if normalize_company(gt_val) == normalize_company(pred_val):
                is_match = True
        elif field == 'metric_name':
            if normalize_metric(gt_val) == normalize_metric(pred_val):
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
            field_stats[field] = {"correct": 1, "total": 1}
        else:
            field_stats[field] = {"correct": 0, "total": 1}
            
    if total_fields == 0:
        return 1.0, 0, {} # No fields to check
        
    return correct_fields / total_fields, total_fields, field_stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against ground truth.")
    parser.add_argument("--gt-files", nargs='+', required=False, help="Paths to ground truth JSONL files.")
    parser.add_argument("--extracted-file", required=False, help="Path to extracted guidance JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Matching threshold.")
    
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
    
    # Per-field metrics
    field_stats = {f: {"correct": 0, "total": 0} for f in COMPARE_FIELDS}
    
    # Iterate over all UIDs in GT
    all_uids = set(gt_data.keys()) | set(extracted_data.keys())
    
    print(f"\nEvaluating {len(all_uids)} documents...")
    
    # Use a loose threshold for alignment to analyze discrepancies
    ALIGNMENT_THRESHOLD = 0.4
    
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
            
        # Item matching with loose threshold for alignment
        matches = match_items(gt_items, pred_items, threshold=ALIGNMENT_THRESHOLD)
        
        for gt, pred, score in matches:
            # Strict Metric Calculation (User's Threshold)
            is_strict_match = False
            if gt and pred and score >= args.threshold:
                is_strict_match = True
                item_metrics["matched"] += 1
            elif gt and not pred:
                item_metrics["missing"] += 1
            elif not gt and pred:
                item_metrics["extra"] += 1
            else:
                # GT and Pred exist but score < args.threshold (Partial Match)
                # For strict metrics, this is a Miss AND an Extra (mismatch)
                item_metrics["missing"] += 1
                item_metrics["extra"] += 1
            
            # Field Analysis (On all aligned pairs)
            if gt and pred:
                acc, count, item_field_stats = score_item(gt, pred)
                if count > 0:
                    field_accuracies.append(acc)
                
                # Update per-field stats
                for f, stats in item_field_stats.items():
                    field_stats[f]["correct"] += stats["correct"]
                    field_stats[f]["total"] += stats["total"]
            
            # For completely missing items (no alignment even at 0.4), we can optionally penalize
            # But the user wants to see "where the mismatch is", so we focus on the aligned ones.
            # If we include unaligned missing items as 0% accuracy, it dilutes the "mismatch" signal.
            # Let's ONLY track field stats for aligned pairs (gt and pred).
                
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
    
    print("\n--- Item Level (Strict Threshold: {args.threshold}) ---")
    print(f"Total GT Items: {item_metrics['total_gt']}")
    print(f"Total Pred Items: {item_metrics['total_pred']}")
    print(f"Matched: {item_metrics['matched']}")
    print(f"Missing (FN): {item_metrics['missing']}")
    print(f"Extra (FP): {item_metrics['extra']}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    print(f"\n--- Field Discrepancy Analysis (Aligned Pairs @ Threshold {ALIGNMENT_THRESHOLD}) ---")
    print(f"Average Field Accuracy (on aligned pairs): {avg_field_accuracy:.2%}")
    
    print("\n--- Per-Field Accuracy (Aligned Pairs Only) ---")
    for f in COMPARE_FIELDS:
        stats = field_stats[f]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{f}: {acc:.2%} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()
