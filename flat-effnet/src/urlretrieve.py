import pandas as pd
import urllib.error
import json
import os
from urllib.request import urlretrieve
from tqdm import tqdm

product_focus = {
    1.0: '2.55',
    2.0: 'BOY',
    3.0: 'GABRIELLE',
    4.0: 'TIMELESS',
    5.0: 'CLASSIC',
    6.0: 'COCOHANDLE',
    7.0: '19',
    8.0: 'OTHER',
}

def train_test_split(df, frac=0.1, seed=1221):
    """
    splits train and test set for each ["product_type_code", "PRODUCT_FOCUS"]
    """
    test = df.groupby(["product_type_code", "PRODUCT_FOCUS"]).sample(frac=frac, random_state=seed)
    df = df.loc[test.index,'set'] = 'test'
    return df

def retrieve(df):
    """
    load dataset imgs from imageurl.csv into folder ./dataset as .jpg
    """
    statusOutput = {}
    img_404 = []
    img_connRefused = []

    print('Retrieving images into /input. . .')
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        try:
            urllib.request.urlopen(row.IMAGE_URL)
        # except urllib.error.HTTPError as e:
        #     # Return code error (e.g. 404, 501, ...)
        #     print("HTTPError: {}".format(e.code))
        #     img_404.append((index + 1, row.id))
        # except urllib.error.URLError as e:
        #     # Not an HTTP-specific error (e.g. connection refused)
        #     print("URLError: {}".format(e.reason))
        #     img_connRefused.append(row.id)
        except:
            print("BaseException")
            img_connRefused.append(row.id)
        else:
            # 200
            categ = product_focus[row.PRODUCT_FOCUS] if row.PRODUCT_FOCUS in product_focus else 'OTHER'
            filename = f'../input/seg_{row.set}/{categ}/{row.id}.jpg'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            urlretrieve(row.IMAGE_URL, f'../input/seg_{row.set}/{categ}/{row.id}.jpg')

    statusOutput["Total Images"] = len(df)
    statusOutput["Invalid Images"] = len(img_404) + len(img_connRefused)
    statusOutput["img_404"] = img_404

    with open("./output.json", "w") as file:
        json.dump(statusOutput, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # df = pd.read_csv("../csv/sample.csv", index_col=[0])
    # sample = train_test_split(df)
    # sample.to_csv('./sample_split.csv')
    bags = pd.read_csv('../csv/bags_split.csv')
    retrieve(bags)
