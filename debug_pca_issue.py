#!/usr/bin/env python3
"""深度诊断PCA维度问题"""

import json
import os
import sys
import glob
import numpy as np

def debug_beauty_embeddings(dataset_name='AmazonReviews2014', category='Beauty'):
    """详细检查Beauty数据集的embedding问题"""
    print("="*80)
    print(f"深度诊断: {dataset_name}/{category} - PCA维度问题")
    print("="*80)

    # 1. 检查数据集基本信息
    dataset_dir = f"cache/{dataset_name}/{category}"
    processed_dir = os.path.join(dataset_dir, "processed")

    if not os.path.exists(processed_dir):
        print(f"❌ processed目录不存在: {processed_dir}")
        return

    # 2. 检查id_mapping
    id_mapping_path = os.path.join(processed_dir, "id_mapping.json")
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            id_mapping = json.load(f)
        n_items = len(id_mapping['id2item'])
        n_users = len(id_mapping['id2user'])
        print(f"\n✓ 数据集统计:")
        print(f"  Items: {n_items}")
        print(f"  Users: {n_users}")
    else:
        print(f"\n❌ id_mapping.json不存在")
        return

    # 3. 检查所有embedding相关文件
    print(f"\n{'='*80}")
    print("检查所有embedding文件:")
    print('='*80)

    # 3.1 检查 *.sent_emb 文件 (sentence embedding after PCA)
    sent_emb_files = glob.glob(os.path.join(processed_dir, "*.sent_emb"))
    print(f"\n📁 Sentence Embedding文件 (*.sent_emb):")
    if sent_emb_files:
        for sent_file in sent_emb_files:
            filename = os.path.basename(sent_file)
            size_mb = os.path.getsize(sent_file) / (1024 * 1024)
            print(f"\n  文件: {filename}")
            print(f"  大小: {size_mb:.2f} MB")

            try:
                # 读取embedding
                sent_embs = np.fromfile(sent_file, dtype=np.float32)
                total_elements = len(sent_embs)
                print(f"  总元素数: {total_elements:,}")

                # 推测shape
                expected_samples = n_items - 1  # 排除padding item
                if total_elements % expected_samples == 0:
                    emb_dim = total_elements // expected_samples
                    print(f"  推测Shape: ({expected_samples}, {emb_dim})")
                    print(f"  每个item的embedding维度: {emb_dim}")

                    # 分析维度问题
                    if emb_dim == 85:
                        print(f"\n  ⚠️⚠️⚠️ 发现问题!")
                        print(f"  Embedding维度是85，但PCA目标是128")
                        print(f"  这就是报错的原因: n_components=128 > min(n_samples, n_features)=85")
                        print(f"\n  可能原因:")
                        print(f"    1. 这是一个旧的cache文件，之前用了--sent_emb_pca=85")
                        print(f"    2. 或者之前PCA就失败了，生成了错误的文件")
                        print(f"\n  解决方案:")
                        print(f"    删除这个文件，让系统重新生成:")
                        print(f"    rm {sent_file}")
                    elif emb_dim == 128:
                        print(f"  ✓ 正确: 已经是PCA后的128维")
                    elif emb_dim == 768:
                        print(f"  ✓ 正确: 原始T5 embedding (768维)")
                    elif emb_dim < 128:
                        print(f"  ⚠️ 维度({emb_dim}) < PCA目标(128)")
                        print(f"  这个文件可能是用不同配置生成的")
                else:
                    print(f"  ⚠️ 无法推测shape (总元素数不能被items数整除)")
                    print(f"  可能的shape组合:")
                    for possible_dim in [768, 128, 85, 64, 32]:
                        if total_elements % possible_dim == 0:
                            possible_samples = total_elements // possible_dim
                            print(f"    - ({possible_samples}, {possible_dim})")

            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
    else:
        print("  ❌ 没有找到*.sent_emb文件")
        print("  说明还没有生成sentence embedding")

    # 3.2 检查 *.sem_ids 文件 (semantic cluster IDs)
    sem_ids_files = glob.glob(os.path.join(processed_dir, "*.sem_ids"))
    print(f"\n📁 Semantic IDs文件 (*.sem_ids):")
    if sem_ids_files:
        for sem_file in sem_ids_files:
            filename = os.path.basename(sem_file)
            size_mb = os.path.getsize(sem_file) / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")
    else:
        print("  ❌ 没有找到*.sem_ids文件")

    # 3.3 检查multimodal相关文件
    multimodal_files = glob.glob(os.path.join(processed_dir, "*multimodal*.npy"))
    if multimodal_files:
        print(f"\n📁 Multimodal文件:")
        for mm_file in multimodal_files:
            filename = os.path.basename(mm_file)
            size_mb = os.path.getsize(mm_file) / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")
            try:
                data = np.load(mm_file)
                print(f"    Shape: {data.shape}")
            except:
                pass

    # 4. 检查metadata文件
    print(f"\n📁 Metadata文件:")
    metadata_files = glob.glob(os.path.join(processed_dir, "metadata.*.json"))
    for meta_file in metadata_files:
        filename = os.path.basename(meta_file)
        size_mb = os.path.getsize(meta_file) / (1024 * 1024)
        print(f"  - {filename} ({size_mb:.2f} MB)")

    # 5. 总结
    print(f"\n{'='*80}")
    print("诊断总结")
    print('='*80)

    # 检查是否有85维的文件
    has_85_dim_issue = False
    if sent_emb_files:
        for sent_file in sent_emb_files:
            sent_embs = np.fromfile(sent_file, dtype=np.float32)
            expected_samples = n_items - 1
            if len(sent_embs) % expected_samples == 0:
                emb_dim = len(sent_embs) // expected_samples
                if emb_dim == 85:
                    has_85_dim_issue = True
                    print(f"\n❌ 发现85维问题!")
                    print(f"   文件: {sent_file}")
                    print(f"\n🔧 修复步骤:")
                    print(f"   1. 删除旧的cache文件:")
                    print(f"      rm {sent_file}")
                    print(f"   2. 重新构建vocabulary:")
                    print(f"      python build_vocab.py --dataset={dataset_name} --category={category}")

    if not has_85_dim_issue:
        if not sent_emb_files:
            print("\n✓ 没有发现85维问题 (但也没有embedding文件)")
            print("  可能还没有运行过vocabulary构建")
        else:
            print("\n✓ 没有发现85维问题")
            print("  Embedding文件看起来正常")

    # 6. 建议的配置
    print(f"\n{'='*80}")
    print("推荐配置")
    print('='*80)
    print(f"\n对于 {dataset_name}/{category} ({n_items} items):")
    print(f"  --sent_emb_pca=128  ✓ (默认，推荐)")
    print(f"\n如果数据集很小 (items < 128):")
    print(f"  --sent_emb_pca=64   或")
    print(f"  --sent_emb_pca=-1   (不降维)")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    # 默认检查Amazon2014 Beauty
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'AmazonReviews2014'
    category = sys.argv[2] if len(sys.argv) > 2 else 'Beauty'

    debug_beauty_embeddings(dataset, category)

    print(f"\n使用方法:")
    print(f"  python debug_pca_issue.py [dataset] [category]")
    print(f"\n示例:")
    print(f"  python debug_pca_issue.py AmazonReviews2014 Beauty")
    print(f"  python debug_pca_issue.py AmazonReviews2018 All_Beauty")
