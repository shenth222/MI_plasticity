# INSERT_YOUR_CODE
import json
import random

def get_random_samples(json_path, sample_num, seed=42):
    """
    从指定json文件加载数据，并随机选取sample_num个样本
    :param json_path: 数据文件路径
    :param sample_num: 选取样本数量
    :param seed: 随机种子
    :return: 样本的列表
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    if sample_num > len(data):
        sample_num = len(data)
    samples = data[:sample_num]
    return samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/data1/shenth/datasets/commonsense/merged_commonsense_train.json", help="数据集json文件路径")
    parser.add_argument("--sample_num", type=int, required=True, help="需要随机抽取的样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（为None则只打印）")
    args = parser.parse_args()

    samples = get_random_samples(args.json_path, args.sample_num, args.seed)

