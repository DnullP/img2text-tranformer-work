import numpy as np


# 计算两个经过标记化的字符串之间的最长公共子序列的长度
def cal_lcs(string, sub):
    """
    string : 经过空格分割的字符串的标记化结果
    sub : 较短的字符串的标记化结果
    """

    # 确保 string 是较长的字符串
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    # 返回该子序列的长度
    return lengths[len(string)][len(sub)]


class Rouge:
    # 初始化
    def __init__(self):
        self.beta = 1.2

    # 计算单个句子
    def calculate_score(self, candidate, refs):
        assert (len(refs) > 0)
        prec = []
        rec = []

        # 将候选句子连接
        token_c_joined = "".join(candidate)
        token_c = token_c_joined.split(" ")

        refs_joined = "".join(refs)

        for reference in [refs_joined]:
            # 按空格分割为单词
            token_r = reference.split(" ")
            # 计算最长公共子序列的长度
            lcs = cal_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        # 若prec_max和rec_max均不为0，则根据 ROUGE-L公式计算得分
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    # 计算给定数据集
    def compute_score(self, gts, res):
        """
        gts: 字典，包含参考句子，键为图像名称，值为标记化的句子列表
        res: 字典，包含候选句子，键为图像名称，值为标记化的句子列表
        """

        # 确保两个字典具有相同的图像名称对应关系
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            score.append(self.calculate_score(hypo, ref))

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)
