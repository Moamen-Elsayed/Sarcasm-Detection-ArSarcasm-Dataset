# Sarcasm-Detection-ArSarcasm-Dataset
## Problem
The literature has a large amount of work on sarcasm and irony detection, which vary from collecting datasets to building detection systems. However, researchers and linguists cannot yet agree on a specific definition of what is considered to be sarcasm. According to (Grice et al., 1975) sarcasm is a form of figurative language where the literal meaning of words is not intended, and the opposite interpretation of the utterance is the intended one. Gibbs Jr et al. (1994) define sarcasm as a bitter and caustic from of irony. According to Merriam Webster’s dictionary 2, sarcasm is “a sharp and often satirical or ironic utterance designed to cut or give pain”, while irony is defined as “ the use of words to express something other than and especially the opposite of the literal meaning”. These definitions are quite close to each other, yet each of them gives a different definition of sarcasm. While most of the literature assumes that sarcasm is a form of irony, Justo et al. (2014) argues that it is not necessarily ironic. Thus, sarcasm is always confused with other forms of figurative language such as metaphor, irony, humour and satire.

## Data
### ArSarcasm-v2 Dataset

**ArSarcasm-v2** is an extension of the original ArSarcasm dataset published along with the paper [From Arabic Sentiment Analysis to Sarcasm Detection: The ArSarcasm Dataset](https://www.aclweb.org/anthology/2020.osact-1.5/). ArSarcasm-v2 conisists of ArSarcasm along with portions of [DAICT corpus](https://www.aclweb.org/anthology/2020.lrec-1.768/) and some new tweets. Each tweet was annotated for sarcasm, sentiment and dialect. The final dataset consists of 15,548 tweets divided into 12,548 training tweets and 3,000 testing tweets. ArSarcasm-v2 was used and released as a part of the [shared task on sarcasm detection and sentiment analysis in Arabic](https://sites.google.com/view/ar-sarcasm-sentiment-detection/)

### Dataset details:
**ArSarcasm-v2** is provided in a CSV format, we provide the same split that was used for the shared task. The training set contains 12,548 tweets, while the test set contains 3,000 tweets.

The dataset contains the following fields:
* `tweet`: the original tweet text.
* `sarcasm`: boolean that indicates whether a tweet is sarcastic or not.
* `sentiment`: the sentiment of the tweet (positive, negative, neutral).
* `dialect`: the dialect used in the tweet, we used the 5 main regions in the Arab world, follows the labels and their meanings:
  * `msa`: modern standard Arabic.
  * `egypt`: the dialect of Egypt and Sudan.
  * `levant`: the Levantine dialect including Palestine, Jordan, Syria and Lebanon.
  * `gulf`: the Gulf countries including Saudi Arabia, UAE, Qatar, Bahrain, Yemen, Oman, Iraq and Kuwait.
  * `magreb`: the North African Arab countries including Algeria, Libya, Tunisia and Morocco.


## Citation

```
@inproceedings
    {abufarha-etal-2021-arsarcasm-v2,
     title = "Overview of the WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic",
     author = "Abu Farha, Ibrahim  and
     Zaghouani, Wajdi  and
     Magdy, Walid",
     booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
     month = april,
     year = "2021",
    }

```
