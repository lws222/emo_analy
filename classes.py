import os
import csv

def folders_to_tsv(neg_dir, pos_dir, output_tsv):
    """
    将 neg / pos 文件夹中的 txt 文件整理为 TSV 文件
    """
    data = []
    idx = 0

    # 处理 neg 文件夹（label = 0）
    for filename in sorted(os.listdir(neg_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(neg_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip().replace("\n", " ")
            data.append([idx, 0, text])
            idx += 1

    # 处理 pos 文件夹（label = 1）
    for filename in sorted(os.listdir(pos_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(pos_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip().replace("\n", " ")
            data.append([idx, 1, text])
            idx += 1

    # 写入 TSV 文件
    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "label", "text"])
        writer.writerows(data)


if __name__ == "__main__":
    neg_folder = ""
    pos_folder = ""
    output_file = "./en_data/train.tsv"

    folders_to_tsv(neg_folder, pos_folder, output_file)
