# Time-Series-Anomaly-Detection-With-Encoders
Applying a unique approach to an existing research paper on rare event detection in time-series data using an encoder-logistic regression ensemble.

The general approach used here is to: Train and tune an autoencoder, extract just the encoder as a custom transformer to get the latent features, lay a logistic regression on top of the pipeline for classification of rare events.


# Rare Event Time-Series Detection In Manufacturing - Yexin Lu, Brandon Hulston
# 1.0 Introduction
The manufacturing industry is highly susceptible to rare events, such as machine failure, which can significantly impact the production process and costs. Machine failure can lead to unexpected downtime, maintenance costs, and loss of revenue. Therefore, preventing such events is a crucial challenge for manufacturing companies looking to optimize their operations and reduce costs. 

In recent years, deep learning techniques have emerged as a promising approach to address this challenge by enabling predictive maintenance, anomaly detection, and fault diagnosis. In this report, we aim to utilize deep learning techniques to predict sensor break events in a paper manufacturing setting. Specifically, we will employ an encoder for dimension reduction and logistic regression for prediction models. By accurately predicting sensor break events, we can prevent costly downtime and improve the overall efficiency of the paper manufacturing process. When a break does happen, it can take a long time to fix and leads to increasingly large losses in profit.

These breaks are often instantaneous events that happen immediately after a bolt breaks. This makes it extremely difficult to capture any meaningful relationships within the data as is. We are hoping that by using an encoding technique, we can extract some unique features/insights from the latent encoding space. In the end, we compare our results to the original paper that uses this dataset with XGBoost, and a logistic regression model.

# 2.0 Dataset
We will be working with a multivariate time series dataset collected from the network of sensors in a pulp-and-paper mill. The dataset contains 18,398 observations of sensors every 2 minutes and includes the following columns:
* time: The timestamp for each observation.
* y: A binary response variable (break:1; not break:0)
* x1-x61: Predictors that are continuous except for x28 and x61. X28 is a categorical variable that denotes the type of paper being produced, while x61 is a binary variable.

This dataset is highly imbalanced, because the failure events occur not frequently. After we grouped data by 5 timesteps, there are only about 4% of the positive (break failure) class. 

# 3.0 Data Preprocessing

## Test Data
First, we obtained the indices of all 1 values for the outcome variable y, and use the last 20 breaks and the corresponding periods before them as our test data. For the test set, we use the same preprocessing method as the training set. 

## Data Cleaning 
We used one-hot encoding to represent the categorical variable 'x28' in our predictors. This technique ensures that each category is treated equally and no unintended biases or relationships are introduced between the categories.
        
## Data Manipulation - time blocks
In order to process the data in a meaningful way that uses previous timesteps as features too, we create time blocks, which are flattened intervals of time as our individual samples. 

For positive instances, we collected all 1's with the previous 5 timesteps and removed them from the training set. We dropped all 1 values and the time and y variables since they are not useful in our model building. We flattened each 5 rows into 1 row, resulting in a dataset of positive samples with 340 columns (68 x 5) per observation. For negative instances, we separated the 0 values rows into groups of 5 continuous time steps. Values are scaled too.

# 4.0 Dataset Creation
Because our task is to build a model to capture extremely low-frequency events (4% of all blocked time data), there are some issues when it comes to building a model that can generalize well and has strong recall. In order to create our new sample, we used resampling techniques to create a bootstrapped sample of the positive time blocks. Using this resampling technique, we created a new set of positives that consisted of 2,000 samples.

<img width="337" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/a24f15ea-e711-4db0-8764-646b0907cb40">

We combined and shuffled this with our other ~3,000 negative samples giving us a dataset with a total of 4,941 samples. We conducted a lot of research to see if resampling could provide better results. One of the main issues we saw is that it can often introduce heavy bias into the models we are working with. However, due to our extremely low positive-negative class ratio and desire to specifically target these positive samples, we concluded that resampling gave us better results in the end.

# 5.0 Modeling Approach
Our approach to this problem consists of a few steps:
* Build an autoencoder function to allow us to easily make custom architectures
* Conduct cross-validation on a variety of parameters to get the best combinations
* Extract just the encoding layers from the best autoencoders
* Build a polynomial logistic regression on top of the encoder to predict breaks

The original paper we got this dataset from (https://arxiv.org/abs/1809.10717) used a suite of complex algorithms and sampling techniques to achieve pretty strong results. In the end though, they were only able to achieve a precision score of 7% and a F-1 of 11%. In contrast, you will see by the end of our report that our approach eventually gives us a 9% precision score, 15% F-1 score, and 37% recall score. 

<img width="376" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/a6a45359-dac9-4ee4-a828-2541e0c3c536">

We primarily measure success by the precision, recall, and F-1 score because this is an imbalanced dataset where we are trying to predict the minority class.

## Building the Autoencoder
Our Autoencoder function can be found in appendix B, but at its core, the function is able to customize the number of layers, neurons in layers, and if batch normalization is turned on:
* Builds a number of predefined Dense layers with decreasing number of neurons to act as an encoder
* Adds an intermediary bottleneck layer which holds our encodings
* Builds the same number of Dense layers with increasing number of neurons to act as our decoder
* With input of 3 layers, 3 encoding layers + 1 bottleneck + 3 decoding layers = 7 layers total

## Cross-Validation
We can use this UDF to run 3-fold cross-validation on our training data. We create nested for loops of all our possible hyper parameters, meaning that in each loop we have a model with unique combinations of parameters. For each model, we then run our cross-validation and take the average of our results on the validation data. This helps us identify the encoders with the best encoding and decoding ability. 

We selected our best 3 combinations which you can see below:

* 3 Dense layers, 256 neurons, bath_norm = False
* 4 Dense layers, 256 neurons, bath_norm = False
* 5 Dense  layers, 256 neurons, bath_norm = False

After running some more tests, it seemed like the more complex model with 5 layers and 256 neurons in the first layer would give us the best combination of generalization and identification power. 

## Encoding Model
In order to create the encoding model, we just need to grab the first half of layers of the model. This can be done as follows:
encoder_model = tf.keras.Model(inputs=auto_encoder3.inputs, outputs=auto_encoder3.layers[len(auto_encoder3.layers) // 2].output)
Which just takes the floor division value of the number of layers in the auto encoder, thus grabbing the first 5 layers in this scenario. 

## Pipeline with Logistic Regression
The last step in building the model is building a logistic regression on top of it. One of the problems that we ran into was telling us that our model was heavily overfitting. The predicted probabilities often came out as [0.999999, 0.000001]. This isn’t acceptable as we want a model with which we can use meaningful thresholds as alerts for potential breaks in the manufacturing.

To handle this, we added a polynomial kernel to the logistic regression with degrees of 2. We also added l2 regularization to it which seemed to improve our generalization as well.

<img width="320" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/c5f223cc-5a26-41a5-ad72-6918721a863d">

In order to actually build this pipeline, we used a custom sklearn transformer for our encoder called ‘PretrainedEncoder’. This transformer used a custom fit method that just returned the model itself since we have already pre-trained our encoder at this point. We would want to avoid retraining when we go to fit the logistic regression on our results. Appendix C code.

# 6.0 Results and Findings

We take our ensemble model and fit it on the testing data which is just a subset of the original data. Then we can perform the logistic regression at several thresholds.

Using a threshold of ~.52, we were able to achieve relatively strong recall, precision, and F-1 scores. When compared to the original paper, we see that we get improvements to our precision and F-1 score as described previously.More impressive is our precision rate which sits at ~37%. 

<img width="280" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/5acda0d4-3e8d-4257-8bab-1bfc66b74719">

We also built a logistic regression but did not add the regularizing features like l2 and the polynomial kernel. We found better results with a simple logistic regression in this case. This model is trained on the raw data instead of the encodings. In comparing the autoencoder to this logistic model, we saw significant improvements in recall, and small improvements in f-1 score. This implies to us that building an encoder can help us understand some underlying features in the latent space of the original data, thus improving our model.

<img width="377" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/dcf875ff-58df-4b78-a753-1c6064cda428">

## Business Application
To translate this into a business context, we are able to catch 37% of all breaks using our encoding method. The downside is that everytime we are notified of a potential break, it will only actually end up breaking every ten times. This is better than an alternative of no alerts though. Having someone double check on it when an alert comes through can significantly reduce losses in profits over time.

In comparison to other approaches, the autoencoder approach was able to have the best results on recall, which is the most important metric in the context of this problem. Being able to identify any potential break before it happens is extremely valuable to the business, and the implications of double-checking the machine every now and then are pretty small.

## What to Improve?
While our results made substantial gains on other methods, we think we could achieve better results given time. One of the biggest challenges was the class imbalance. While imbalance is bad, the bigger issue is the lack of positive samples we have in this dataset. If we were able to collect more data, we would likely have more positive samples to train the model with. Another approach to this problem might help. Using a Variational Autoencoder might help create more healthily distributed positive samples which would help us in identifying these rare events. If we had more time, we would have also liked to try more models to make predictions on the encoding layer such as XGBoost, an SVM, or more densely connected layers.

# 6.0 Bibliography
Original Paper and Dataset: https://arxiv.org/abs/1809.10717
Autoencoders: https://machinelearningmastery.com/autoencoder-for-classification/

# 7.0 Appendix

Appendix A - flattening multiple timesteps into one row:

<img width="330" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/b1dafe4f-ef00-4a92-a2fb-d839e095c0f3">

Appendix B - Autoencoder:

<img width="349" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/3bcbe669-6954-4953-b053-26f02fde3063">

Appendix C - Custom Transformer:

<img width="520" alt="image" src="https://github.com/bhulston/Time-Series-Anomaly-Detection/assets/79114425/98dff969-cb81-4cfb-9161-2ae098e09c65">



