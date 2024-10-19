import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def doc2vec_embedding(df_dir):
    target_description_df = pd.read_excel(df_dir)
    targets = list(target_description_df['Target'])
    target_texts = list(target_description_df['Target_info'])
    documents = [TaggedDocument(doc, [t]) for doc, t in zip(target_texts, targets)]
    model = Doc2Vec(vector_size=2000, min_count=2, epochs=400)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
