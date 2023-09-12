# Data-Driven Price Discovery for New Product Listings on Amazon

## Introduction

The seller’s decision about what price to charge is a critical component of the market mechanism, especially in a competitive market with a large number of sellers. Sellers face a tradeoff between increasing demand and increasing unit profit of a product. Charging too high a price will lead to low demand for the seller’s goods as customers substitute the seller’s goods with the competitors’ more affordable goods. On the other hand, charging too low a price will lead to suboptimal profit for the seller by leaving money on the table. The pricing decision is more challenging for a new seller. On the one hand, there are far more competitors and consumers in the online marketplace than offline, and pricing information is more transparent (Cavallo, 2018). This makes it really difficult for sellers to earn profits based on asymmetric information. On the other hand, they lack experience with operating in the market and information about the strategies followed by their competitors. It may take a long time for them to figure out the best pricing strategy. In this project, we aim to predict a “suggested price” for such a new seller to aid them in their decision to come up with their first posted price on the platform.

## Methods

### Data

We used the Amazon review and product data published by Ni et al. (2019). This is a rich
dataset consisting of 233 million reviews across all products on Amazon.com between May
1996 and Oct 2018. Each review contains the overall rating by the user, upvotes garnered
by the review, a reviewer id, the text of the review, a time stamp, and a summary consisting
of the keywords in the review. There is metadata for each review consisting of product
information which includes the color (white or black), size (large or small), package type
(hardcover or electronics), etc. Product images uploaded by users are also included. Finally,
there is product metadata for 15.5 million products which includes a description of the
features of the product, along with crucially, the price in US dollars and the sales rank of
the product within its product category.

Our final dataset consists of products in the Tools and Home Improvement sub-category
under Home and Kitchen category. Due to space constraints, we started with the Electronics
dataset but we had a bottleneck due to limited RAM space while calculating the clusters.
So, we had to use a smaller subset of the dataset.
Another reason for choosing this sub-category is that products in this category tend to be
highly homogeneous. Moreover, the subjective aesthetic component in this category should
be of less importance than in categories like fashion, toys, and games, etc. We expect people
to care about the description for the Home and Kitchen category and make their product
choices depending on what the sellers have said, compared to categories like beauty products
where customers may use Instagram or Youtube to choose products instead of descriptions.
The third reason is that the Home and Kitchen category has a large number of small
sellers. There is a limited number of big brands with large market power to fix prices in
this category due to the low barrier to entry arising from less technological complexity in
manufacturing.

Our final dataset consists of products in the Tools and Home Improvement sub-category
under Home and Kitchen category. Due to space constraints, we started with the Electronics
dataset but we had a bottleneck due to limited RAM space while calculating the clusters.
So, we had to use a smaller subset of the dataset.
Another reason for choosing this sub-category is that products in this category tend to be
highly homogeneous. Moreover, the subjective aesthetic component in this category should
be of less importance than in categories like fashion, toys, and games, etc. We expect people
to care about the description for the Home and Kitchen category and make their product
choices depending on what the sellers have said, compared to categories like beauty products
where customers may use Instagram or Youtube to choose products instead of descriptions.
The third reason is that the Home and Kitchen category has a large number of small
sellers. There is a limited number of big brands with large market power to fix prices in
this category due to the low barrier to entry arising from less technological complexity in
manufacturing.

### Model

We used 3 models—K-means clustering, Random Forest, and XGBoost—to predict the
prices. Our first benchmark aimed to use the information contained in product titles, in
particular the similarity between product titles to make sub-category groups. We used noun
chunks, or noun phrases in linguistics, to vectorize each title instead of words, the com-
monly used unit in a document, since (1) it makes sense to use only nouns when it comes
to the product titles listed on online shopping platforms and (2) noun chunks retain im-
portant information of the product features. For example, instead of treating “Wireless”,
“Bluetooth”, “Headphones”, and “Earbuds” as distinct features when inferring the similar-
ity between products, we used a noun chunk, “Wireless Bluetooth Headphones Earbuds” as
a feature (see Table 1, for the examples of noun chunks). The groups computed using the
similarity between product titles could then be used to calculate the representative price for
similar products in each group. We used K-means clustering for this task because it provides
an efficient way to cluster without making a lot of assumptions about the relationships be-
tween underlying variables. For each cluster, we calculated the average within-cluster price
as Benchmark 1. The clusters computed in this stage are used as an input feature for later
analyses.

We decided to use the Random Forests regression because we are making an inference
about prices of unseen products and wanted to use an ensemble method which can decrease
variance at the expense of training accuracy. We initially got low accuracy but reducing the
max depth and increasing the n estimators (using GridSearchCV) led to large improvements in accuracy. Another reason in favor of using Random Forests is lower space and
processing requirements than neural networks and other resource-heavy techniques. We also
implemented another ensemble method—XGBoost regressor. The advantage of XGBoost is
that it converges more quickly with fewer steps and has lower compute costs.
The input features we use are as follows:
1. Brand (categorical): brand names, excluding those where there are less than 1000 product items listed
2. N images (numeric): the number of images the vendor included in the product description section
3. HighResImg (numeric): binary feature, 1, implying that the vendor included a high resolution image, 0, not
4. N description (numeric): length of product descriptions, calculated by the number of words in the description of each product.
5. Cluster (categorical): the group, or the sub-category of the product, computed using the K-means clustering method with the product titles


For preprocessing before using Random Forests and XGBoost, we used MinMaxScaler
for our three numeric variables, N images, HighResImg, and N description; and one-hot
encoding for categorical variables, brand and cluster computed using the title similarity.

![image](https://github.com/seuraha/pricing_ml/assets/71260667/b19972a3-6a60-4a5a-88c7-5d51f9f79ddb)

### Code
`Metadata cleaning Dec 2.ipynb`

This file works with the raw data within Home and Kitchen category. We first converted the
json formatted data into Python dataframe (pandas) while dropping unnecessary features
like what products the customers also bought or also viewed. We also converted string type
features, such as the URL of images in the product description section, into numeric features and cleaned the HTML tags and special characters in the descriptions. As the next step, we
cleaned the price data since they were in string type with “$” as the prefix. The observations
with no price data were around 1.4%, and we dropped these observations. The number of
brands in the data was 58, 687, which can be inefficient memory-wise when converted into
one-hot vectors. Most of the brands (87.44%) have less than 1, 000 listed products, so we
dropped those brands and ended up with 37 brands. Finally, we restricted our sample to
those in Tools & Home Improvement sub-category which has 11, 062 products.

`Title similarity home kitchen Dec2.ipynb`

This file works with only the title data and tries to find the product similarities. We first
vectorized the title of each product using noun chunks. We then applied K-means clustering
to categorize products into clusters and computed the evaluation metrics for both training
and test data using the mean price within cluster.
This step also generates sub-categories that will be used in later analysis. The average
prices within clusters are also used as our Benchmark 1.

![image](https://github.com/seuraha/pricing_ml/assets/71260667/919156a8-2f42-4f9f-b6e2-e14f2f0ea448)

`Benchmark2 home kitchen Combined.ipynb`

For Benchmarks 2 and 3, we first performed the preprocessing and train-test split. Then we
applied the two models respectively and use GridSearchCV to find best parameters.
The preprocessing stage for two models is the same. For categorical variables, we
performed one-hot encoding before splitting the data since the training and test datasets
must have the same features. Then we split the data product-level, and built a pipeline.
The pipeline included scaling numeric variables using MinMaxScaler() and the regressor, (RandomForestRegressor or XGBRegressor). We fit the training data using GridSearchCV
to find out the best model.

## Evaluation
We used R2, mean squared error, and mean absolute error for evaluation. We split the
training and test data with sizes 0.75 and 0.25, respectively. The train-test split was product-
level. The following table summarizes the evaluation metrics on test data.

![image](https://github.com/seuraha/pricing_ml/assets/71260667/66f338e6-ead3-458e-a89a-c03f24c36660)

Using XGBoost gave us the best model with the highest R2, the lowest MSE (mean square
error), and the lowest MAE (mean absolute error). Across evaluation metrics, the accuracy
increased from the simple averaging model to the K-means clustering to the Random forest
and was the highest in the XGBoost model. This trend also holds for increasing model
complexity and sophisticated ensembling techniques too.

## Related Work
There is a rich literature on pricing by sellers and price discovery in the context of the online
marketplace (see Gerpott and Berends (2022) for a recent review). Auction mechanisms were
important in the earlier days of the Internet but recently, “posted prices” have become the
dominant method for pricing (Einav et al., 2018). A strand of literature deals with dynamic
algorithmic pricing by sellers using near real-time data on their competitors (Chen et al.,
2016; Elmaghraby and Keskinocak, 2003; Garbarino and Lee, 2003). However, this is only
optimal for companies that have large scales, rich data, and strong bargaining power. An
online retailer who is new to the platform is unlikely to have access to information about
their competitors or other knowledge of the marketplace.

## Conclusion

Setting first prices on e-business platforms can be challenging for new sellers. In this project,
we suggest prices to new sellers based on product descriptions and prices of similar products.
We used 3 models—K-means clustering, Random Forest, and XGBoost—to generate the
suggestions.

Though our analysis did not generate highly accurate predictions, we still think our
research question is worth investigating, and what we have done can be a good starting
point. The greatest challenge for us is to work with Big data in terms of space and computing resources. We started out with a 11GB dataset of all product descriptions for the Electronics
category on Amazon, but encountered difficulties due to the limited memory size while trying
to perform TF-IDF for the title similarity and one-hot encoding for categorical variables.
We ended up trying out models with only a small sub-category in the Home and Kitchen
category. Moreover, we were not able to test and include more features because of the
computing power limitations. We believe that we would get more accurate results if we
could include more useful features and try out different models.

In our future work, we would like to add more features about the reviews, including the
number of reviews of competing vendors, the overall rating of the reviews, and a set of features
that covers the quality of product reviews of competing vendors. A high number/quality
of reviews implies that the competing vendors have already established a reputation in the
market. A new vendor then might need to offer lower prices to differentiate themselves
from competing vendors. Adding those features will give us more information about the
competition in the market.

To have a real-world evaluation metric, we can also conduct a field experiment. This
would involve asking a group of sellers who want to sell their products but have not yet set
a price on Amazon. We can request them to share the title and description of their new
product with us and use that to calculate a predicted price range using our model. Finally,
we can show these new sellers our predicted price range and elicit their acceptance of this
as an evaluation metric.
