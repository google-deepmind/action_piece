#!/usr/bin/env python3
"""诊断Amazon 2014 vs 2018数据集差异"""

import json
import os
import sys

def check_dataset(dataset_name, category):
    """检查特定数据集和类别"""
    print(f"\n{'='*70}")
    print(f"检查数据集: {dataset_name} / {category}")
    print('='*70)

    # 检查目录
    dataset_dir = f"cache/{dataset_name}/{category}"
    if not os.path.exists(dataset_dir):
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return None

    print(f"✓ 找到数据集目录: {dataset_dir}\n")

    # 检查processed文件
    processed_dir = os.path.join(dataset_dir, "processed")
    if not os.path.exists(processed_dir):
        print(f"❌ processed目录不存在")
        return None

    # 检查关键文件
    id_mapping_path = os.path.join(processed_dir, "id_mapping.json")
    if not os.path.exists(id_mapping_path):
        print(f"❌ id_mapping.json不存在 - 数据集尚未处理")
        print(f"\n提示：请先运行数据预处理:")
        print(f"  python build_vocab.py --dataset={dataset_name} --category={category}")
        return None

    # 读取数据统计
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    n_items = len(id_mapping['id2item'])
    n_users = len(id_mapping['id2user'])

    print(f"数据集统计:")
    print(f"  - Items: {n_items:,}")
    print(f"  - Users: {n_users:,}")

    # 检查sentence embedding
    import glob
    sent_emb_pattern = os.path.join(processed_dir, "*.sent_emb")
    sent_emb_files = glob.glob(sent_emb_pattern)

    if sent_emb_files:
        print(f"\n✓ 找到sentence embedding文件:")
        for sent_file in sent_emb_files:
            filename = os.path.basename(sent_file)
            size_mb = os.path.getsize(sent_file) / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")

            # 读取并检查维度
            try:
                import numpy as np
                sent_embs = np.fromfile(sent_file, dtype=np.float32)

                expected_samples = n_items - 1
                total_elements = len(sent_embs)

                if total_elements % expected_samples == 0:
                    emb_dim = total_elements // expected_samples
                    print(f"    Shape: ({expected_samples}, {emb_dim})")

                    if emb_dim < 128:
                        print(f"    ⚠️  维度({emb_dim}) < PCA目标(128) - 无法降维到128!")
                        print(f"    解决方案: 设置 --sent_emb_pca={emb_dim}")
                    elif emb_dim == 128:
                        print(f"    ✓ 已经是PCA后的128维")
                    elif emb_dim == 768:
                        print(f"    ✓ 原始T5 embedding (768维)")
            except Exception as e:
                print(f"    ❌ 读取失败: {e}")

    return {
        'n_items': n_items,
        'n_users': n_users,
        'processed': True
    }


def main():
    print("="*70)
    print("Amazon Reviews 数据集诊断工具")
    print("="*70)

    # 检查2014 Beauty
    result_2014 = check_dataset('AmazonReviews2014', 'Beauty')

    # 检查2018 All_Beauty (注意名称不同!)
    result_2018 = check_dataset('AmazonReviews2018', 'All_Beauty')

    # 总结
    print(f"\n{'='*70}")
    print("总结与建议")
    print('='*70)

    print("\n📌 关键差异:")
    print("  - Amazon2014: 类别名为 'Beauty'")
    print("  - Amazon2018: 类别名为 'All_Beauty' ⚠️ 名称不同!")

    if result_2014 and result_2014['processed']:
        print(f"\n✅ Amazon2014/Beauty 已处理:")
        print(f"   - {result_2014['n_items']:,} items")
        print(f"   - 可以直接训练")

    if not result_2018 or not result_2018.get('processed'):
        print(f"\n❌ Amazon2018/All_Beauty 未处理:")
        print(f"   请运行:")
        print(f"   python build_vocab.py --dataset=AmazonReviews2018 --category=All_Beauty")

    print("\n" + "="*70)
    print("常见类别名称对照表")
    print("="*70)
    print(f"{'2014':<30} {'2018':<30}")
    print("-"*70)
    print(f"{'Beauty':<30} {'All_Beauty':<30}")
    print(f"{'CDs_and_Vinyl':<30} {'CDs_and_Vinyl':<30}")
    print(f"{'Sports_and_Outdoors':<30} {'Sports_and_Outdoors':<30}")
    print("="*70)

    # 如果用户报错是85维问题
    print("\n📋 关于 PCA n_components=85 错误:")
    print("这个错误通常发生在:")
    print("  1. 数据集很小（items < 128）")
    print("  2. Sentence embedding已经被降维到85维")
    print("\n解决方案:")
    print("  1. 检查数据集是否已下载并处理完成")
    print("  2. 如果数据集确实很小，降低PCA维度:")
    print("     --sent_emb_pca=64 (或者更小的值)")
    print("  3. 使用更大的数据集 (如 Sports_and_Outdoors, CDs_and_Vinyl)")

if __name__ == '__main__':
    main()
