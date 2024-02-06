import pandas as pd
from preprocess import TextPreprocessor

class HateXplain:
    def __init__(self, data_path):
        self.data_path = data_path
        self.text_processor = TextPreprocessor()

    def preprocess_data(self):
        df = pd.read_json(self.data_path)
        df_t = df.T
        
        # Extracting each label from nested list
        df_t['annotators0'] = df_t['annotators'].apply(lambda x: x[0]['label'])
        df_t['annotators1'] = df_t['annotators'].apply(lambda x: x[1]['label'])
        df_t['annotators2'] = df_t['annotators'].apply(lambda x: x[2]['label'])

        # Labelling
        df_t['annotators0'] = df_t['annotators0'].apply(lambda x: 0.0 if x == "hatespeech" else (1.0 if x == "offensive" else 2.0))
        df_t['annotators1'] = df_t['annotators1'].apply(lambda x: 0.0 if x == "hatespeech" else (1.0 if x == "offensive" else 2.0))
        df_t['annotators2'] = df_t['annotators2'].apply(lambda x: 0.0 if x == "hatespeech" else (1.0 if x == "offensive" else 2.0))

        # Dropping some unnecessary columns
        df_t.drop(['post_id', 'annotators'], axis=1, inplace=True)

        # Separating non_toxic
        df_n = df_t[((df_t['annotators0'] == 2.0) & (df_t['annotators1'] == 2.0) & (df_t['annotators2'] == 2.0)) | (df_t['rationales'].apply(lambda x: len(x) == 0))]
        df_n['label'] = 'non_toxic'

        # Generating rationales values for non-rationales by dividing 1 by the total length
        df_n['leng'] = df_n['post_tokens'].apply(lambda x: len(x))
        df_n['rationales'] = df_n['leng'].apply(lambda length: [1 / length] * length)

        # Separating toxic
        df_ho = df_t[~df_t.index.isin(df_n.index)]

        # Taking mean of rationales
        df_ho['rationales1'] = df_ho['rationales'].apply(lambda x: x[0])
        df_ho['rationales2'] = df_ho['rationales'].apply(lambda x: x[1])
        df_ho['rationales3'] = df_ho['rationales'].apply(lambda x: x[2:3])

        df_ho['sum_result'] = df_ho.apply(lambda row: [a + b + c for a, b, c in zip(row['rationales1'], row['rationales2'], row['rationales1'])], axis=1)
        df_ho['mean'] = df_ho['sum_result'].apply(lambda x: [val / 3 for val in x])

        df_toxic = df_ho[['post_tokens', 'mean']].rename(columns={'mean': 'rationales'})
        df_toxic['label'] = 'toxic'

        df_combine = pd.concat([df_toxic, df_n[['post_tokens', 'label', 'rationales']]]).reset_index(drop=True)

        df_combine["post_tokens"].values
        for i in range(len(df_combine["post_tokens"])):
            df_combine["post_tokens"][i] = " ".join(df_combine["post_tokens"][i])

        df_combine["post_tokens"] = df_combine["post_tokens"].apply(lambda x: self.text_processor.text_preprocessing(x))

        df_combine.rename(columns={"post_tokens": "tweet", "label": "class"}, inplace=True)

        return df_combine