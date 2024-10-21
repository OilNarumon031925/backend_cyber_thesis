"""
    This method is use to helpe preparing the data or clean of Machine Learning Model

"""

import pandas as pd
from src.helper.preparing import remove_punctuation, remove_stopword
from pythainlp.tokenize import word_tokenize
import numpy as np

def prepare(text: str, vectorizer) -> pd.DataFrame:
    """
     The prepare method is used for preparing text as a input of model
    """

    try:
        
        df = pd.DataFrame({
            "data": [text]
        })

        df['clean'] = df.data.apply(remove_punctuation)
        df['clean'] = df['clean'].apply(word_tokenize)
        df['clean'] = df['clean'].apply(remove_stopword)

        _vector_trans = vectorizer.transform(df['clean'])
        array_tfidf = np.array(_vector_trans.todense())
        return pd.DataFrame(array_tfidf, columns=vectorizer.get_feature_names_out())
    
    except Exception as e:
        raise ValueError("Can't prepare the data", e)