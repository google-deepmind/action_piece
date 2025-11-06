#!/usr/bin/env python3
"""诊断Beauty数据集的PCA维度问题"""

import json
import os
import sys

def check_beauty_dataset():
    """检查Beauty数据集的基本信息"""
    print("=" * 60)
    print("诊断 Beauty 数据集")
    print("=" * 60)

    # 1. 检查数据集是否存在
    beauty_dir = "cache/AmazonReviews2014/Beauty"
    if not os.path.exists(beauty_dir):
        print(f"❌ Beauty数据集目录不存在: {beauty_dir}")
        print("需要先运行训练来下载和处理数据")
        return

    print(f"✓ 找到Beauty数据集目录: {beauty_dir}\n")

    # 2. 检查id_mapping（看有多少items）
    id_mapping_path = os.path.join(beauty_dir, "processed/id_mapping.json")
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            id_mapping = json.load(f)

        n_items = len(id_mapping['id2item'])
        n_users = len(id_mapping['id2user'])
        print(f"数据集统计:")
        print(f"  - Items数量: {n_items}")
        print(f"  - Users数量: {n_users}")
        print()
    else:
        print(f"❌ 找不到id_mapping.json")
        n_items = None

    # 3. 检查sentence embedding缓存
    processed_dir = os.path.join(beauty_dir, "processed")
    if os.path.exists(processed_dir):
        print(f"processed目录下的文件:")
        for file in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
        print()

    # 4. 检查sent_emb缓存文件（看实际生成了多少维）
    import glob
    sent_emb_files = glob.glob(os.path.join(processed_dir, "sent_emb_*.dat"))
    if sent_emb_files:
        print(f"找到 {len(sent_emb_files)} 个sentence embedding缓存文件:")
        for sent_file in sent_emb_files:
            print(f"  - {os.path.basename(sent_file)}")

            # 尝试读取并查看维度
            try:
                import numpy as np
                sent_embs = np.fromfile(sent_file, dtype=np.float32)

                # sent_embs应该是 (n_items-1) x embedding_dim
                if n_items:
                    expected_samples = n_items - 1
                    total_elements = len(sent_embs)

                    # 推测embedding维度
                    if total_elements % expected_samples == 0:
                        emb_dim = total_elements // expected_samples
                        print(f"    Shape: ({expected_samples}, {emb_dim})")
                        print(f"    Total elements: {total_elements}")

                        # 检查是否是PCA之后的结果
                        if emb_dim == 128:
                            print(f"    ⚠️  已经是PCA后的128维！")
                        elif emb_dim == 768:
                            print(f"    ✓ 原始T5 embedding 768维")
                        elif emb_dim < 128:
                            print(f"    ⚠️  维度({emb_dim}) < PCA目标维度(128)!")
                            print(f"    这就是问题所在：数据集太小，无法降维到128维")
                    else:
                        print(f"    ⚠️  无法整除，可能数据不匹配")
                        print(f"    Total elements: {total_elements}, expected samples: {expected_samples}")
            except Exception as e:
                print(f"    ❌ 读取失败: {e}")
        print()
    else:
        print("❌ 没有找到sentence embedding缓存文件")
        print()

    # 5. 分析问题
    print("=" * 60)
    print("问题分析:")
    print("=" * 60)
    print("PCA报错: n_components=128 must be between 0 and min(n_samples, n_features)=85")
    print()
    print("可能的原因:")
    print("1. Beauty数据集只有85个items（太小）")
    print("2. 或者sentence embedding已经被降维到85维")
    print()

    if n_items and n_items <= 100:
        print(f"⚠️  Beauty数据集确实很小 (只有{n_items}个items)")
        print(f"   无法使用PCA降维到128维")
        print()
        print("解决方案:")
        print("1. 使用更大的数据集 (如 Sports_and_Outdoors, CDs_and_Vinyl)")
        print("2. 或者修改配置，降低sent_emb_pca维度:")
        print(f"   --sent_emb_pca={min(85, n_items-1)}")

    print("=" * 60)

if __name__ == '__main__':
    check_beauty_dataset()
