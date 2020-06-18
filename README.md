---


---

<h1 id="amazon-movie-star-rating-prediction">Amazon Movie Star Rating Prediction</h1>
<h2 id="introduction">Introduction</h2>
<p>This project examines the realtionship between the star rating score between user reviews from Amazon Movie Reviews using the available features. The features include unique identifier for the product/user, the number of users who found the review helpful, the number of users who indicated whether they found the review helpful, the timestamp for the review, the brief summary of the review and the text of the review. In the rest of this notebook, I’m going to utilize these features on the prediction of the star rating score. This project, if successful, is beneficial to estimate any unlisted movie’s popularity or reputation, which could be further used in the recommendation system in this field. For any further detail about this competition, please refer  <a href="http://www.kaggle.com/c/bu-cs506-spring-2020-midterm">here</a>.</p>
<h2 id="eda">EDA</h2>
<p>There are 9 columns in total within this dataset. “Id” is the identifier of each record row, therefore apparently it is relevant to our modeling or prediction.<br>
Apart from that “ProductId” and “UserId” are categorical features while the others are numerical. So it requires me to process them in different ways. Next I’m going to take an in-depth look into them one by one.</p>
<h3 id="user-id--product-id">User ID &amp; Product ID</h3>
<p>There are more than 1 million records inside the dataset. Neither the most active users nor the most rated products have more than 3000 entries. In other words they barely make a distortion to our modeling. So I can just keep them as they are.<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/users.png" alt="enter image description here"><br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/products.png" alt="enter image description here"></p>
<h3 id="helpfulness-numerator--denominator">Helpfulness Numerator &amp; Denominator</h3>
<p>Similarly, according the following visualization, none of a certain value of either of these two features makes up too big a portion, say, more than a half. Thus it is safe to put them as they are except doing some necessary standardization.<br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/helpfulLabels.png" alt="enter image description here"><br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/commentsWatched.png" alt="enter image description here"></p>
<h3 id="score">Score</h3>
<p>Unlike previous features, the rating of 5/5 is too overwhelming because the users were likely to rate a single movie by default. This might cause distortion to our prediction because a predictor inclines to set an unknown record as the most commonly seen value (according to Maximum Likelihood Theorem). This is absolutely we do not want to see. I am going to avoid this happens in the rest of my project.<br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/Scores.png" alt="enter image description here"></p>
<h3 id="summary--text">Summary &amp; Text</h3>
<p>Ideally, I want to utilize all available columns in the dataset. However it is not feasible if any two of them are too strongly correlated with each other. So I make a rough judgement here printing out a word cloud of the text features: Summary and Text.<br>
As it is seen, they have little keywords in common, even though it is partly because that Summary is more consise and contains fewer information. Anyway, I assume they are acceptably indepedendent from each other.<br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/wordCloud.png" alt="enter image description here"></p>
<h3 id="missing-values">Missing values</h3>
<p>According the overview suggests, there are missing values inside the dataset. And a visualization is good option to show where they are and what we should do to impute.<br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/missing.png" alt="enter image description here"><br>
As the figure above illustrates, most empty values exist in the column Score. This is normal because score is the output and the training/testing set are mixed together.<br>
As for other missing values, I chose heat map to pick them out.<br>
<img src="https://github.com/MemphisMeng/CS506-Midterm/blob/master/images/missingHeatmap.png" alt="enter image description here"><br>
Obviously they are barely correlated. This allows me to just apply a simply method to impute without worrying about introducing too many noises.</p>
<h3 id="eda-summary">EDA Summary:</h3>
<ul>
<li>In Score, the distribution of each values are too imbalanced, requiring me to rectify it.</li>
<li>Summary and Text are seemingly independent with each other, I am going to put both of them in use. I will also discuss further about this later.</li>
<li>Other features are good and can be kept as they are.</li>
</ul>
<h2 id="data-preprocessing">Data Preprocessing</h2>
<h3 id="categorical-features">Categorical features:</h3>
<p>Since they are of string type, it is necessary to digitalize them. The reason I chose one-hot encoding over label encoding is due to the fact that label encoding labels each unique identifier a different on a basis of “first appear, first encode”. In other words, the identifier which appears earlier gets a smaller value. This would conflict with the fact that identifier cannot be measureable feature. The only disvantage of one-hot encoding, on the other hand, is that it consumes too many memories. I will show how to handle this situation.</p>
<pre><code>OHE = OneHotEncoder(sparse=True)
IDs = OHE.fit_transform(data[['ProductId', 'UserId']])
</code></pre>
<h3 id="numerical-features">Numerical features:</h3>
<p>Standardization helps to reduce the influence of outliers and converge faster.</p>
<pre><code>data['Helpful'] = data['HelpfulnessNumerator']
data['Unhelpful'] = data['HelpfulnessDenominator'] - data['HelpfulnessNumerator']
scaler = StandardScaler()
data[['Helpful', 'Unhelpful', 'Time']] =
scaler.fit_transform(data[['Helpful', 'Unhelpful', 'Time']])
data = data.drop(['HelpfulnessDenominator','HelpfulnessNumerator'], axis=1)
</code></pre>
<h3 id="missing-value-imputation">Missing value imputation:</h3>
<pre><code>data['Text'].loc[data['Text'].isna()] = ''
data['Summary'].loc[data['Summary'].isna()] = ''
</code></pre>
<h3 id="summary-and-text">Summary and Text:</h3>
<p>In order to extract useful information from these two columns, I applied TF/IDF on them. In other words, I vectorized them so that they can be treated as numerical values.</p>
<pre><code>    text_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')
summary_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english')
text_matrix = text_vectorizer.fit_transform(data['Text'])
summary_matrix = summary_vectorizer.fit_transform(data['Summary'])
</code></pre>
<h3 id="data-preprocessing-summary">Data Preprocessing Summary:</h3>
<p>I have done the following things in this stage:</p>
<ul>
<li>One-hot encoded the categorical values</li>
<li>standardized the numerical values</li>
<li>vectorized the plain text features based on TF/IDF</li>
<li>Imputed missing values</li>
</ul>
<h2 id="modeling">Modeling</h2>
<p>I chose three models that can potentially model the relation between the inputs and the output:</p>
<ul>
<li>Logistics Regression: an essential methodology in the field of classification</li>
<li>Decision Tree: this simple algorithm is ideal for multiclass prediction</li>
<li>Random Forest: essentially creates multiple trees, and averages their results<br>
I have summarized the results (highest accuracy within the validation) in the following table:</li>
</ul>

<table>
<thead>
<tr>
<th>Model</th>
<th>RMSE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Logistics Regression</td>
<td>0.733</td>
</tr>
<tr>
<td>Decision Tree</td>
<td>1.48</td>
</tr>
<tr>
<td>Random Forest</td>
<td>1.52</td>
</tr>
</tbody>
</table><p>So Logistics Regression is the best predictor based on my observation. I took the classifier whose performance is the best to predict the given test data and got an accuracy of approximately 0.80. I ranked No.2 (top 3%) in this competition.</p>
<h2 id="discussion">Discussion</h2>
<p>I used two tricks in this project. Firstly in terms of HelpfulnessDenominator and HelpfulnessNumerator, I found the first one always cover the latter which cause colinearity or something like that. I replaced them with two other columns: “Helpful” and “Helpless” representing the number of people who think a comment is helpful or not.<br>
Another trick is that I implemented K-fold validation when modeling. This makes my comparation between different models more reasonable because they took all data for both training and testing, reducing the bias.</p>
<p>Nevertheless there are a few things where a huge space for improvement lays. Even though I judged Summary and Text are barely related to each other, I have not had carried out suficient proof for that. This is because there are many features after I implemented TFIDF vectors, and when I plotted a heat map it required too many RAMs and caused my session to break down. If it is possible, I will implement PCA or Truncated SVD on them to reduce unnecessary information.<br>
Moreover, because Random Forest and Decision Tree is too time-consuming and I pruned a tree before it was mature. So chances are that the model are underfitted. Perhaps when it fully develops we would have a better predictor.</p>
<h2 id="conclusion">Conclusion</h2>
<p>To sum up, I have developed a model that is predicting a movie on Amazon’s streaming platform based on the reviews from the audience. After performing the exploratory data analysis, I fitted three models and compared their accuracy scores. The model performing the best was the logistics regression model. I reused the model on the given test dataset and achieved an accuracy of 0.80, which has beaten 97% of my competitors.</p>

