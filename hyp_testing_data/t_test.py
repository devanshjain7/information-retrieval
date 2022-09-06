from scipy import stats
import json

sample1 = json.load(open("sampled_ndcg_naive_vsm.txt", "r"))
sample2 = json.load(open("sampled_ndcg_spell_correct_new_token.txt", "r"))
sample3 = json.load(open("sampled_ndcg_lsa.txt", "r"))
sample4 = json.load(open("sampled_ndcg_query_exp.txt", "r"))
sample5 = json.load(open("sampled_ndcg_word2vec.txt", "r"))

samples = [sample2, sample3, sample4, sample5]
models = ["VSM with corrections", "LSA", "Query Expansion", "Word2Vec"]

for model, sample in zip(models, samples):
    print(f"t test results for Baseline vs {model}")
    print(stats.ttest_rel(sample1, sample, alternative='less'))
    print("")
