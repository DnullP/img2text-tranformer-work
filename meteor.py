import nltk
from nltk.translate import meteor_score
import numpy as np
nltk.download('wordnet')
nltk.download('omw-1.4')


class Meteor:
    def __init__(self):
        pass

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            hypo_joined = "".join(hypo)
            ref_joined = "".join(ref)

            tests = hypo_joined.split(" ")
            refs = ref_joined.split(" ")

            score.append(meteor_score.meteor_score([refs], tests))

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

