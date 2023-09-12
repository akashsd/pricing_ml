def metadata_cleaning(filename):
    import pandas as pd
    import json
    import gzip
    import re

    """
    Load the gzip format metadata and return a Pandas DataFrame
    - while dropping unnecessary features,
    - converting string type features such as the url of images in the product description into numeric features, and
    - cleaning HTML tags and special characters in the descriptions
    - dropping observations with no price data (N=8741, 1.4% of the original data)
    - cleaning the brand data as the number of brands in the data was 58,687, 
    with most of them (87.44%) have less than 1000 products listed. The resulting dataset includes a total of 37 brands,
    which was one-hot encoded

    """

    f = gzip.open(filename)
    l = []

    for idx, item in enumerate(f):
        item = json.loads(item)
        if item.get('price') != "": # if price data exists
            # dropping unnecessary features
            item.pop('also_buy', None)
            item.pop('also_view', None)
            item.pop('fit', None)
            item.pop('similar_item', None)
            item['N_images'] = len(item.get('imageURL', []))
            item['HighResImg'] = 1 if item.get('imageURLHighRes', "") != "" else 0
            item.pop('imageURL', None)
            item.pop('imageURLHighRes', None)
            item['N_description'] = len(re.sub("((\<\w+\>)|(\<\/\w+\>)|\s{2,}|\t)", "", " ".join(item['description'])).split())
            item.pop('description', None)
            item.pop('details', None)
            item.pop('tech1', None)
            item.pop('tech2', None)
            item.pop('rank', None)
            item.pop('feature', None)
            item.pop('date', None)
            
            item['main_cat'] = re.sub("(amp\;)", "", item['main_cat']) # main category
            l.append(item)

    data = pd.DataFrame.from_dict(l)

    # price data
    data.loc[:, 'price'] = data.loc[:, 'price'].apply(lambda x: x.replace('$', '').split()[0])
    data.loc[:, 'price'] = pd.to_numeric(data.loc[:, 'price'], errors='coerce')
    data = data.dropna() # drop observations with no price data 

    # brand data
    who_knows = pd.DataFrame(data.brand.value_counts()[data.brand.value_counts() < 1000]).reset_index().loc[:,'index'].to_list()
    data.loc[:, 'brand'] = data.loc[:, 'brand'].apply(lambda x: x if x not in who_knows else "")

    return data