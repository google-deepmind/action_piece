#!/usr/bin/env python3
"""Diagnose metadata coverage issues in Amazon datasets.

This script analyzes which items are missing metadata and provides statistics
to help determine if the coverage rate is normal or indicates a data problem.

Usage:
    python scripts/diagnose_metadata_coverage.py \
        --dataset AmazonReviews2018 \
        --category CDs_and_Vinyl
"""

import argparse
import json
import os
from collections import defaultdict


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_metadata_coverage(cache_dir, dataset, category):
    """Analyze metadata coverage statistics."""

    # Paths
    base_path = os.path.join(cache_dir, dataset, category, 'processed')
    id_mapping_file = os.path.join(base_path, 'id_mapping.json')
    metadata_file = os.path.join(base_path, 'metadata.sentence.json')
    all_item_seqs_file = os.path.join(base_path, 'all_item_seqs.json')

    # Load data
    print(f"Loading data from {base_path}...")
    id_mapping = load_json(id_mapping_file)
    item2meta = load_json(metadata_file)
    all_item_seqs = load_json(all_item_seqs_file)

    # Statistics
    total_items = len(id_mapping['id2item'])
    items_with_meta = len(item2meta)
    items_without_meta = total_items - items_with_meta

    print("\n" + "="*70)
    print("METADATA COVERAGE ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Category: {category}")
    print(f"\nTotal items (from reviews): {total_items}")
    print(f"Items with metadata: {items_with_meta} ({items_with_meta/total_items*100:.2f}%)")
    print(f"Items without metadata: {items_without_meta} ({items_without_meta/total_items*100:.2f}%)")

    # Find missing items
    missing_asins = []
    for i in range(len(id_mapping['id2item'])):
        asin = id_mapping['id2item'][i]
        if asin not in item2meta:
            missing_asins.append(asin)

    # Analyze ASIN format distribution
    numeric_missing = sum(1 for asin in missing_asins if asin.isdigit())
    alphanumeric_missing = len(missing_asins) - numeric_missing

    print(f"\n{'Missing Items by ASIN Format':-^70}")
    print(f"Numeric ASINs (ISBN-10): {numeric_missing} ({numeric_missing/len(missing_asins)*100:.1f}%)")
    print(f"Alphanumeric ASINs: {alphanumeric_missing} ({alphanumeric_missing/len(missing_asins)*100:.1f}%)")

    # Analyze interaction frequency of missing items
    missing_interaction_counts = defaultdict(int)
    for user, items in all_item_seqs.items():
        for item in items:
            if item in missing_asins:
                missing_interaction_counts[item] += 1

    if missing_interaction_counts:
        interaction_counts = list(missing_interaction_counts.values())
        print(f"\n{'Interaction Statistics for Missing Items':-^70}")
        print(f"Total interactions with missing items: {sum(interaction_counts)}")
        print(f"Average interactions per missing item: {sum(interaction_counts)/len(interaction_counts):.2f}")
        print(f"Min interactions: {min(interaction_counts)}")
        print(f"Max interactions: {max(interaction_counts)}")

        # Distribution
        bins = [1, 5, 10, 20, 50, 100, 1000]
        dist = defaultdict(int)
        for count in interaction_counts:
            for i, threshold in enumerate(bins):
                if count < threshold:
                    dist[f"<{threshold}"] += 1
                    break
            else:
                dist[">=1000"] += 1

        print(f"\nInteraction count distribution:")
        for range_label in [f"<{b}" for b in bins] + [">=1000"]:
            if range_label in dist:
                count = dist[range_label]
                print(f"  {range_label:>8}: {count:>5} items ({count/len(missing_asins)*100:>5.1f}%)")

    # Sample missing ASINs
    print(f"\n{'Sample Missing ASINs':-^70}")
    print("First 20 missing ASINs:")
    for i, asin in enumerate(missing_asins[:20], 1):
        asin_type = "ISBN-10" if asin.isdigit() else "Amazon"
        interactions = missing_interaction_counts.get(asin, 0)
        print(f"  {i:2}. {asin} ({asin_type:>6}) - {interactions} interactions")

    # Assessment
    print(f"\n{'ASSESSMENT':-^70}")
    coverage_pct = items_with_meta / total_items * 100

    if coverage_pct >= 98:
        status = "✅ EXCELLENT"
        comment = "Coverage is very high, no concerns."
    elif coverage_pct >= 95:
        status = "✅ GOOD"
        comment = "Coverage is within normal range for Amazon datasets."
    elif coverage_pct >= 90:
        status = "⚠️  ACCEPTABLE"
        comment = "Coverage is slightly low but still usable. Common for categories like CDs, Clothing."
    elif coverage_pct >= 85:
        status = "⚠️  BORDERLINE"
        comment = "Coverage is low. Consider investigating data quality or using alternative metadata sources."
    else:
        status = "❌ POOR"
        comment = "Coverage is very low. Data may be corrupted or incomplete."

    print(f"Status: {status}")
    print(f"Comment: {comment}")

    # Recommendations
    print(f"\n{'RECOMMENDATIONS':-^70}")
    if coverage_pct < 95:
        print("1. Check if you downloaded the complete metadata file")
        print("2. Verify the metadata file is not corrupted (try re-downloading)")
        print("3. Consider using metadata='none' mode if coverage is too low")
        print("4. For research purposes, current coverage may be acceptable")
    else:
        print("No action needed. Current coverage is sufficient for training.")

    print("="*70 + "\n")

    return {
        'total_items': total_items,
        'items_with_meta': items_with_meta,
        'coverage_pct': coverage_pct,
        'missing_asins': missing_asins
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose metadata coverage')
    parser.add_argument('--dataset', default='AmazonReviews2018',
                        help='Dataset name')
    parser.add_argument('--category', default='CDs_and_Vinyl',
                        help='Category name')
    parser.add_argument('--cache_dir', default='./cache',
                        help='Cache directory')

    args = parser.parse_args()

    analyze_metadata_coverage(args.cache_dir, args.dataset, args.category)


if __name__ == '__main__':
    main()
