from sentence_transformers import SentenceTransformer, util
import logging


def match_topics_to_opinions(topics_df, opinions_df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    topic_embeddings = model.encode(topics_df['text'].tolist(), convert_to_tensor=True)
    opinion_embeddings = model.encode(opinions_df['text'].tolist(), convert_to_tensor=True)

    matched_opinions = {}
    for i, topic_embedding in enumerate(topic_embeddings):
        logging.info(f"Processing topic {i + 1}/{len(topics_df)}")
        cosine_scores = util.pytorch_cos_sim(topic_embedding, opinion_embeddings)
        top_opinions = [opinions_df.iloc[int(idx)]['text'] for idx in cosine_scores[0].topk(5).indices]
        matched_opinions[topics_df.iloc[i]['text']] = top_opinions

    return matched_opinions
