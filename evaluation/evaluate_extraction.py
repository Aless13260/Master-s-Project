import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import statistics
import sys

import re

GT_DIR = Path("evaluation") / "ground_truth"
GT_PATH = [f for f in GT_DIR.iterdir() if f.is_file()]
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
]

def normalize_string(s):
    if not s:
        return ""
    s = str(s).strip().lower()
    # Replace smart quotes
    s = s.replace('’', "'").replace('“', '"').replace('”', '"')
    return s

def normalize_ticker(t):
    """Normalize ticker to uppercase."""
    if not t:
        return ""
    return str(t).upper().strip()

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
    if "total expenses" in s or "operating expense" in s:
        return "opex"
        
    return s

def normalize_period(p):
    """
    Normalize reporting period for comparison purposes.
    This handles format variations so equivalent periods match.
    
    e.g. 'FY25' -> 'fy2025'
    'Q1 2025' -> 'q1 fy2025'
    'FY 2023' -> 'fy2023'
    'FY2025-FY2027' -> 'fy2025-2027'
    '2025' -> 'fy2025'
    'Q4' -> 'q4' (can't add FY without context)
    'fourth quarter' -> 'q4'
    'next quarter' -> 'next quarter' (relative, can't normalize without context)
    """
    s = normalize_string(p)
    if not s:
        return s
    
    # Convert spelled-out quarters to Qn format
    quarter_map = {
        'first quarter': 'q1',
        'second quarter': 'q2', 
        'third quarter': 'q3',
        'fourth quarter': 'q4',
        '1st quarter': 'q1',
        '2nd quarter': 'q2',
        '3rd quarter': 'q3',
        '4th quarter': 'q4',
    }
    for spelled, qn in quarter_map.items():
        if spelled in s:
            s = s.replace(spelled, qn)
    
    # Convert "full year" to "fy" (will get year attached below if present)
    s = s.replace('full year', 'fy')
    s = s.replace('fiscal years', 'fy')  # plural form for ranges
    s = s.replace('fiscal year', 'fy')
    
    # Remove spaces in FY
    s = s.replace('fy ', 'fy')
    
    # Handle "fiscal years 2026 through 2028" -> "fy2026-2028"
    m = re.search(r'fy\s*(\d{4})\s*(?:through|to|-)\s*(?:fy\s*)?(\d{4})', s)
    if m:
        return f"fy{m.group(1)}-{m.group(2)}"
    
    # Handle standalone year: "2025" -> "fy2025"  
    if re.match(r'^20\d{2}$', s):
        return f"fy{s}"
    
    # Replace FY25 with FY2025
    s = re.sub(r'fy(\d{2})\b', lambda m: f'fy20{m.group(1)}', s)
    
    # Handle "Q1 2025" or "Q1 FY2025" -> "q1 fy2025"
    # First, ensure year has FY prefix
    s = re.sub(r'\b(q\d)\s+(?:fy)?(\d{4})\b', r'\1 fy\2', s)
    
    # Handle ranges like FY2025-FY2027 -> fy2025-2027
    parts = s.split('-')
    if len(parts) == 2:
        p1 = parts[0].strip()
        p2 = parts[1].strip()
        
        # If p2 is just '27', make it '2027'
        if p2.isdigit() and len(p2) == 2:
            p2 = "20" + p2
        elif p2.isdigit() and len(p2) == 4:
            pass  # already full year
            
        # Normalize to: FY2025-2027
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
    Filters out items with guidance_type='other' as they are secondary/ambiguous.
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
            
            # Filter out guidance_type='other' items (secondary/ambiguous)
            guidance_items = [
                item for item in guidance_items 
                if item.get('guidance_type') and item.get('guidance_type').lower() != 'other'
            ]
            
            gt_by_uid[uid] = guidance_items
    return gt_by_uid

def load_extracted_data(extracted_path):
    """
    Load extracted data.
    Supports both one guidance item per line (dict) and grouped items per line (list).
    Returns a dictionary mapping uid -> list of guidance items.
    Filters out items with guidance_type='other' as they are secondary/ambiguous.
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
                # Filter out guidance_type='other' items
                filtered = [
                    item for item in guidance_data
                    if item.get('guidance_type') and item.get('guidance_type').lower() != 'other'
                ]
                extracted_by_uid[uid].extend(filtered)
            else:
                # Single item - check if it should be included
                if guidance_data.get('guidance_type') and guidance_data.get('guidance_type').lower() != 'other':
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

def score_item(gt, pred, track_field=None):
    """
    Calculate per-field accuracy for a matched pair.
    Returns: (accuracy, total_fields, field_stats_dict, mismatch_info)
    
    If track_field is specified, mismatch_info will contain details for that field.
    """
    if not gt:
        # Extra item (False Positive)
        return 0.0, 0, {}, None
        
    if not pred:
        # Missing item (False Negative)
        # Should not happen here if called correctly, but for safety
        return 0.0, len(COMPARE_FIELDS), {}, None
    
    correct_fields = 0
    total_fields = 0
    field_stats = {}
    mismatch_info = None
    
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
            gt_normalized = normalize_period(gt_val)
            pred_normalized = normalize_period(pred_val)
            if gt_normalized == pred_normalized:
                is_match = True
            elif track_field == 'reporting_period':
                mismatch_info = {
                    'gt_raw': gt_val,
                    'pred_raw': pred_val,
                    'gt_normalized': gt_normalized,
                    'pred_normalized': pred_normalized,
                    'ticker': gt.get('ticker'),
                    'metric': gt.get('metric_name'),
                }
        elif field == 'ticker':
            if normalize_ticker(gt_val) == normalize_ticker(pred_val):
                is_match = True
        elif field == 'metric_name':
            if normalize_metric(gt_val) == normalize_metric(pred_val):
                is_match = True
        elif field == 'unit':
            gt_normalized = normalize_unit(gt_val)
            pred_normalized = normalize_unit(pred_val)
            if gt_normalized == pred_normalized:
                is_match = True
            elif track_field == 'unit':
                mismatch_info = {
                    'gt_raw': gt_val,
                    'pred_raw': pred_val,
                    'gt_normalized': gt_normalized,
                    'pred_normalized': pred_normalized,
                    'ticker': gt.get('ticker'),
                    'metric': gt.get('metric_name'),
                }
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
            elif track_field == field:
                mismatch_info = {
                    'gt_raw': gt_val,
                    'pred_raw': pred_val,
                    'gt_normalized': normalize_string(gt_val),
                    'pred_normalized': normalize_string(pred_val),
                    'ticker': gt.get('ticker'),
                    'metric': gt.get('metric_name'),
                }
                
        if is_match:
            correct_fields += 1
            field_stats[field] = {"correct": 1, "total": 1}
        else:
            field_stats[field] = {"correct": 0, "total": 1}
            
    if total_fields == 0:
        return 1.0, 0, {}, None # No fields to check
        
    return correct_fields / total_fields, total_fields, field_stats, mismatch_info

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against ground truth.")
    parser.add_argument("--gt-files", nargs='+', required=False, help="Paths to ground truth JSONL files.")
    parser.add_argument("--extracted-file", required=False, help="Path to extracted guidance JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Matching threshold.")
    parser.add_argument("--show-mismatches", type=str, default=None, help="Show mismatched values for a specific field (e.g., 'reporting_period').")
    parser.add_argument("--no-synthetic", action="store_true", help="Exclude synthetic ground truth data (synthtic_guidance.jsonl).")
    parser.add_argument("--synthetic-only", action="store_true", help="Use only synthetic ground truth data.")
    parser.add_argument("--show-extras", action="store_true", help="Show extra predicted items (false positives) that don't match any GT.")
    
    args = parser.parse_args()
    
    # Determine which GT files to load
    if args.gt_files:
        gt_paths = [Path(p) for p in args.gt_files]
    else:
        gt_paths = GT_PATH
        # Filter based on synthetic flags
        if args.no_synthetic:
            gt_paths = [p for p in gt_paths if 'synth' not in p.name.lower()]
        elif args.synthetic_only:
            gt_paths = [p for p in gt_paths if 'synth' in p.name.lower()]
    
    # Load data
    gt_data = load_ground_truth(gt_paths)
    if args.extracted_file:
        extracted_data = load_extracted_data(args.extracted_file)
    else:
        extracted_data = load_extracted_data(TEST_PATH)
    
    # Filter extracted data to only include UIDs present in ground truth
    extracted_data = {uid: items for uid, items in extracted_data.items() if uid in gt_data}
    
    # Metrics
    doc_metrics = {"total": 0, "with_guidance_gt": 0, "with_guidance_pred": 0, "correct_detection": 0}
    item_metrics = {"total_gt": 0, "total_pred": 0, "matched": 0, "missing": 0, "extra": 0}
    field_accuracies = []
    
    # Per-field metrics
    field_stats = {f: {"correct": 0, "total": 0} for f in COMPARE_FIELDS}
    
    # Track mismatches for the specified field
    field_mismatches = []
    
    # Track extra predictions (false positives)
    extra_items = []
    
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
                extra_items.append({'uid': uid, 'item': pred})
            else:
                # GT and Pred exist but score < args.threshold (Partial Match)
                # For strict metrics, this is a Miss AND an Extra (mismatch)
                item_metrics["missing"] += 1
                item_metrics["extra"] += 1
            
            # Field Analysis (On all aligned pairs)
            if gt and pred:
                acc, count, item_field_stats, mismatch_info = score_item(gt, pred, track_field=args.show_mismatches)
                if count > 0:
                    field_accuracies.append(acc)
                
                # Track mismatches for the specified field
                if mismatch_info:
                    mismatch_info['uid'] = uid
                    field_mismatches.append(mismatch_info)
                
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
    
    print(f"\n--- Item Level (Strict Threshold: {args.threshold}) ---")
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
    
    # Print mismatches for the specified field
    if args.show_mismatches and field_mismatches:
        print(f"\n--- Mismatches for '{args.show_mismatches}' ({len(field_mismatches)} total) ---")
        for i, m in enumerate(field_mismatches, 1):
            print(f"{i}. {m.get('ticker', 'Unknown')} - {m.get('metric', 'Unknown')}")
            print(f"   GT:   {m.get('gt_raw')}")
            print(f"   Pred: {m.get('pred_raw')}")
    
    # Print extra predictions (false positives)
    if args.show_extras and extra_items:
        print(f"\n--- Extra Predictions (False Positives): {len(extra_items)} ---")
        for i, e in enumerate(extra_items, 1):
            item = e['item']
            print(f"{i}. {item.get('ticker', 'Unknown')} - {item.get('metric_name', 'Unknown')}")
            print(f"   Type: {item.get('guidance_type')} | Period: {item.get('reporting_period')}")
            print(f"   Values: low={item.get('guided_range_low')}, high={item.get('guided_range_high')}, current={item.get('current_value')}")

if __name__ == "__main__":
    main()
