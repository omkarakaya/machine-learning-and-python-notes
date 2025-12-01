- We create a list of evaluation methods of LLMs for a given use case.
- Use case: Generate 5 questions per document from a given set of documents.

# Evaluation Based on Gold Data
- The evaluation dataset has questions per document.
- We want to evaluate if the generated questions are similar to the ones
in the evaluation dataset.

## Blue
    `Geometric mean of Precision of ngrams * Brevity Penalty`
- Brevity Penalty: `exp(1 - reference_length / candidate_length)`
- Brevity Penalty is 1 if the prediction length is equal to 
or longer than the reference length.
- The disadvantage of BLEU is that it does not consider importance of the
words and meaning.

## ROUGE
    `Sum of Recall of ngrams`
- Rough has similar problems as BLEU.

## BERTScore
- BERTScore is an automatic evaluation metric for text 
generation that computes a similarity score for each token 
in the candidate sentence with each token in the reference sentence.

We can match generated questions to the ones in the gold dataset
by using the metrics above. Then, we can calculate the final score
per document.

# Evaluation without Gold Data
- We can create a rubric of evaluation criteria:
    - Gramatically correctness: Is the question gramatically correct.
    - Clarity: Are the questions clear.
    - Answer availability: Are the answer available in the context.
- We can also create an additional rubric for the set of questions:
    - Coverage: Do the questions cover the main points in the documents.
    - Diversity: Are the questions about different topics.
- We need to set a point (1-10) for each question and item in the rubric pair.
- We can do it either by using llm-as-a-judge or human evaluation.
- Fundamentals of this approach is based on [G-Eval framework](https://arxiv.org/pdf/2303.16634)
## Rubric - Human Evaluation
- We can hire human evaluators to set a point to the questions.
- One disadvantage is that we will need human evaluation we create an 
evaluation dataset and it's costly. As an example, we may generate questions by using multiple LLMs, change prompt etc. 
Each change will lead to a different evaluation dataset.
- If we are have too many items to evaluate in the test set, we may choose to sample it 
with some margin of error.
    - We can derive a formula from the Central Limit Theorem that can calculate
number of items to evaluate to get a similar score to the population which is the complete testset.
    - We should also consider stratified sampling based on the labels of the 
    documents/questions such as domain, year, length etc.
    - Diversity the samples based on embeddings spaces or/and clustering.
    
## Rubric - LLM as a Judge
- Rather than Human Evaluation, we can choose to use LLM-as-a-judge.
- We aim to mimic the human evaluation by using LLMs.
- The main idea is to tune a prompt on a small subset of the test set:
    - Pick ~100 representative items from the test set.
    - Evaluate with human evaluation.
    - Tune a prompt which achieves a similar evaulation performance to metrics from the users.
    - We can use traditional train-validation-test splits for prompt tuninig.
    - Use the prompt on larger tasks.
- We can rely on one or multiple LLMs.
- If we use multiple LLMs, we can apply majority voting or averages.


# Pairwise Comparison
- 

# Ranking Based Comparison
- 




# Perplexity
    perplexity = e**(sum(losses) / num_tokenized_tokens)
- Given a model and an input text sequence, perplexity measures how likely the 
model is to generate the input text sequence.
- As a metric, it can be used to evaluate how well 
the model has learned the distribution of the text it was trained on.

