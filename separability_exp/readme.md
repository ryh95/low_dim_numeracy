separability has been done by others

try clustering separability

since the number of word embeddings is huge, we use sampling

1. sample 10% numbers and 10% words, use kernel kmeans, kernel is that used in the separability test, see paper
,result: about 0.02 nmi, similar with kmeans
2. sample words, number of the sampled words is the same with the 'numbers', result: about 0.28 nmi, kmeans is about 0.43
3. random sample some words from embeddings as 'numbers', the number is tha same with real numbers, random sample
some other words from rest of embeddings as 'words', the number is the same with real numbers, kernel kmeans clustering result: about 0,
kmeans is also about 0

poor performance may due to the class imbalance(numbers are far less than non numeric words) 