from rouge import Rouge
from cider import Cider
from meteor import Meteor


# DeepFashion数据集评分类
class DeepFashionEvalCap:
    def __init__(self, df, dfRes):
        """
        :param df: 参考句子， 字典类型， key为图片id， value为句子列表
        :param dfRes: 测试句子， 字典类型， key为图片id， value为句子列表
        """
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.df = df
        self.dfRes = dfRes
        self.params = {'image_id': df.keys()}

    def evaluate(self):
        image_ids = self.params['image_id']
        gts = {}
        res = {}

        for image_id in image_ids:
            gts[image_id] = self.df[image_id]
            res[image_id] = self.dfRes[image_id]

        # 标记化
        gts = self.tokenize(gts)
        res = self.tokenize(res)

        # 设置评分器
        print('Setting up scorers...')
        scorers = [
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE'),
            (Cider(), 'CIDER')
        ]

        # 计算分数
        for scorer, method in scorers:
            print('Calculating %s score...' % method)
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
                    print('%s: %0.3f' % (m, sc))
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
                print('%s: %0.3f' % (method, score))
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, image_ids, method):
        for image_id, score in zip(image_ids, scores):
            if image_id not in self.imgToEval:
                self.imgToEval[image_id] = {}
                self.imgToEval[image_id]['image_id'] = image_id
            self.imgToEval[image_id][method] = score

    def set_eval_imgs(self):
        self.evalImgs = [eval for _, eval in self.imgToEval.items()]

    # 设置标记化为静态方法
    @staticmethod
    def tokenize(annotations):
        return {image_id: [ann for ann in anns] for image_id, anns in annotations.items()}
