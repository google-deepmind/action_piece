#!/usr/bin/env python3
"""检查metadata覆盖率"""

import json
import os
import sys

def check_metadata_coverage(dataset_name='AmazonReviews2014', category='Beauty'):
    """检查有多少items有metadata"""
    print("="*80)
    print(f"检查Metadata覆盖率: {dataset_name}/{category}")
    print("="*80)

    processed_dir = f"cache/{dataset_name}/{category}/processed"

    # 1. 读取id_mapping
    id_mapping_path = os.path.join(processed_dir, "id_mapping.json")
    if not os.path.exists(id_mapping_path):
        print(f"❌ {id_mapping_path} 不存在")
        return

    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    n_items = len(id_mapping['id2item'])
    print(f"\n总Items数: {n_items}")
    print(f"ASINs: {id_mapping['id2item'][:5]}... (前5个)")

    # 2. 读取metadata
    metadata_files = [
        'metadata.sentence.json',
        'metadata.raw.json',
    ]

    for meta_filename in metadata_files:
        meta_path = os.path.join(processed_dir, meta_filename)
        if os.path.exists(meta_path):
            print(f"\n{'='*80}")
            print(f"检查: {meta_filename}")
            print('='*80)

            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            print(f"Metadata条目数: {len(metadata)}")

            # 检查有多少items有metadata
            items_with_meta = 0
            items_without_meta = 0
            missing_asins = []

            for item_id in range(1, n_items):  # 跳过padding item (id=0)
                asin = id_mapping['id2item'][item_id]
                if asin in metadata and metadata[asin]:
                    items_with_meta += 1
                else:
                    items_without_meta += 1
                    if len(missing_asins) < 10:
                        missing_asins.append(asin)

            coverage = items_with_meta / (n_items - 1) * 100

            print(f"\n统计:")
            print(f"  有metadata的items: {items_with_meta}/{n_items-1} ({coverage:.2f}%)")
            print(f"  缺失metadata的items: {items_without_meta}/{n_items-1} ({100-coverage:.2f}%)")

            if items_without_meta > 0:
                print(f"\n缺失metadata的ASIN示例 (前10个):")
                for asin in missing_asins:
                    print(f"    {asin}")

            # 检查metadata内容
            if metadata:
                sample_asin = list(metadata.keys())[0]
                sample_meta = metadata[sample_asin]
                print(f"\nMetadata示例 (ASIN: {sample_asin}):")
                if isinstance(sample_meta, str):
                    print(f"  类型: string")
                    print(f"  内容: {sample_meta[:200]}...")
                elif isinstance(sample_meta, dict):
                    print(f"  类型: dict")
                    print(f"  Keys: {list(sample_meta.keys())}")
                    for key in list(sample_meta.keys())[:3]:
                        value = sample_meta[key]
                        if isinstance(value, str):
                            print(f"    {key}: {value[:100]}...")
                        else:
                            print(f"    {key}: {value}")

    # 3. 检查是否是sentence类型的问题
    print(f"\n{'='*80}")
    print("可能的问题诊断")
    print('='*80)

    meta_sentence_path = os.path.join(processed_dir, "metadata.sentence.json")
    if os.path.exists(meta_sentence_path):
        with open(meta_sentence_path, 'r') as f:
            metadata_sentence = json.load(f)

        # 检查有多少非空的sentence
        non_empty_sentences = 0
        empty_sentences = 0
        for asin, sentence in metadata_sentence.items():
            if sentence and sentence.strip():
                non_empty_sentences += 1
            else:
                empty_sentences += 1

        print(f"\nMetadata.sentence.json分析:")
        print(f"  总条目: {len(metadata_sentence)}")
        print(f"  非空sentence: {non_empty_sentences}")
        print(f"  空sentence: {empty_sentences}")

        if non_empty_sentences < 100:
            print(f"\n⚠️⚠️⚠️ 发现问题!")
            print(f"  只有{non_empty_sentences}个非空的sentence")
            print(f"  这可能就是为什么PCA只有{non_empty_sentences}个样本的原因!")
            print(f"\n可能的原因:")
            print(f"  1. Metadata文件损坏或格式错误")
            print(f"  2. Metadata下载不完整")
            print(f"  3. 数据预处理出错")
            print(f"\n建议:")
            print(f"  1. 删除processed目录，重新下载和处理:")
            print(f"     rm -rf cache/{dataset_name}/{category}/processed")
            print(f"     python build_vocab.py --dataset={dataset_name} --category={category}")

if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'AmazonReviews2014'
    category = sys.argv[2] if len(sys.argv) > 2 else 'Beauty'

    check_metadata_coverage(dataset, category)
