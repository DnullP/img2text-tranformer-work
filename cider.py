import copy
from collections import defaultdict
import numpy as np
import math


# 预处理
# 将输入的字符串转变为n-gram的形式
def pre_deal(s, n=4, out=False):
    """
    s: sentence, str
    n: n-gram, int
    """
    words = s.split()
    # 词频字典
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


# 处理参考句子
def deal_refs(refs, n=4):
    """
    refs: reference sentence, str
    n: n-gram, int
    """
    return [pre_deal(refs, n)]


# 处理测试句子
def deal_test(test, n=4):
    return pre_deal(test, n, True)


# 定义CIDEr评分类
class CiderScorer(object):

    # 复制句子
    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    # 初始化
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.deal_append(test, refs)
        self.ref_len = None

    # 添加句子
    def deal_append(self, test, refs):
        if refs is not None:
            self.crefs.append(deal_refs(refs))
            if test is not None:
                self.ctest.append(deal_test(test))
            else:
                self.ctest.append(None)

    # 验证数据数量的一致性
    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    # 实现实例的累加
    def __iadd__(self, other):
        if type(other) is tuple:
            # 避免创造新的实例
            self.deal_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    # 计算参考数据的词频，便于后续计算IDF
    def calculate_doc_freq(self):
        for refs in self.crefs:
            # set中的元素不重复
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def calculate_cider(self):
        # 将 n-gram 的计数映射为 TF-IDF 权重的向量
        def counts2vec(cnts):
            # 存储 n-gram 和对应的 TF-IDF 权重
            vec = [defaultdict(float) for _ in range(self.n)]
            # 存储 n-gram 的长度
            length = 0
            # 存储 n-gram 的范数
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # 若ngram 在文档中不存在，则设置文档频率为 1.0
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram)-1
                # 将 TF-IDF 权重计算结果赋值给 vec[n][ngram]
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # 计算向量的范数
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            # 返回向量、范数和 n-gram 的长度
            return vec, norm, length

        # 计算两个向量的余弦相似度
        def cosine_sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            vec_hyp: 与假设（hypothesis）对应的向量
            vec_ref: 与参考（reference）对应的向量
            norm_hyp: 假设向量的范数
            norm_ref: 参考向量的范数
            length_hyp: 假设的长度
            length_ref: 参考的长度
            """
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert (not math.isnan(val[n]))
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # 计算参考长度的对数值
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += cosine_sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            # 平均化分数
            score_avg /= len(refs)
            # 将分数乘以10以便缩放
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def calculate_score(self, option=None, verbose=0):
        # 计算IDF
        self.calculate_doc_freq()
        assert (len(self.ctest) >= max(self.document_frequency.values()))
        # 计算cider分数
        score = self.calculate_cider()
        return np.mean(np.array(score)), np.array(score)


class Cider:

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            hypo_joined = "".join(hypo)
            ref_joined = "".join(ref)

            cider_scorer += (hypo_joined, ref_joined)

        (score, scores) = cider_scorer.calculate_score()

        return score, scores
