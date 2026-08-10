# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# adomlans's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/3393979033.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement AdaDelta Update Step | Implement a vectorized AdaDelta update in NumPy using running gradient and parameter-update averages without a manual learning rate. | https://www.tensortonic.com/problems/adadelta-optimizer |
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Adjusted Cosine Similarity | Compute adjusted cosine similarity for item-based collaborative filtering using co-rated users and mean-centered ratings. | https://www.tensortonic.com/problems/adjusted-cosine-similarity |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Angle Between 3D Vectors | Compute the angle between two 3D vectors in NumPy with clamped cosine values and safe handling of zero norms. | https://www.tensortonic.com/problems/angle-between-3d |
| Compute AUC (Area Under ROC) | Calculate binary-classification ROC AUC from false-positive and true-positive rates using trapezoidal integration. | https://www.tensortonic.com/problems/auc |
| Autocorrelation | Compute normalized time-series autocorrelation across a requested range of lags, including constant-series handling. | https://www.tensortonic.com/problems/autocorrelation |
| Average Pooling 2D | Apply non-overlapping 2D average pooling to rectangular feature maps while discarding incomplete edge windows. | https://www.tensortonic.com/problems/average-pooling-2d |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Baseline Predictor | Predict collaborative-filtering ratings from the global mean plus user and item rating biases. | https://www.tensortonic.com/problems/baseline-predictor |
| Batch Shuffling & Mini-Batch Generator | Create shuffled mini-batches from NumPy feature and target arrays with reproducible ordering and final-batch handling. | https://www.tensortonic.com/problems/batch-generator |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Bigram Probabilities (Add-1 Smoothing) | Estimate bigram probabilities from token sequences using add-one smoothing over a fixed vocabulary. | https://www.tensortonic.com/problems/bigram-probabilities |
| Bilinear Interpolation | Resize a 2D image with bilinear interpolation by combining the four neighboring pixel values at each output coordinate. | https://www.tensortonic.com/problems/bilinear-interpolation |
| Binary Focal Loss | Compute binary focal loss from predicted probabilities with class balancing, focusing strength, and stable logarithms. | https://www.tensortonic.com/problems/binary-focal-loss |
| Binning | Assign numeric values to ordered bins using supplied boundaries while handling values at interval edges. | https://www.tensortonic.com/problems/binning |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| BLEU Score | Calculate a BLEU translation score from candidate and reference tokens using clipped n-gram precision and brevity penalty. | https://www.tensortonic.com/problems/bleu-score |
| Implement BM25 Ranking Score | Implement BM25 document ranking with term frequency saturation, inverse document frequency, and length normalization. | https://www.tensortonic.com/problems/bm25 |
| Bootstrap Mean & Confidence Interval | Estimate a sample mean and confidence interval through reproducible bootstrap resampling of numeric observations. | https://www.tensortonic.com/problems/bootstrap-mean |
| Catalog Coverage | Measure recommendation catalog coverage as the fraction of available items appearing across user recommendation lists. | https://www.tensortonic.com/problems/catalog-coverage |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Chi-Square Test | Run a chi-square independence test on a contingency table using expected counts and the chi-square statistic. | https://www.tensortonic.com/problems/chi2-independence |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Cohen's Kappa | Calculate Cohen's kappa from two label sequences by comparing observed agreement with chance agreement. | https://www.tensortonic.com/problems/cohens-kappa |
| Color to Grayscale | Convert an RGB image to grayscale using weighted color channels while preserving its spatial dimensions. | https://www.tensortonic.com/problems/color-to-grayscale |
| Compute Confusion Matrix with Normalization | Build a multiclass confusion matrix and optionally normalize counts by true-class rows or predicted-class columns. | https://www.tensortonic.com/problems/confusion-matrix-norm |
| Implement Contrastive Loss (Siamese) | Implement Siamese-network contrastive loss using pair labels, embedding distances, and a separation margin. | https://www.tensortonic.com/problems/contrastive-loss |
| 2D Convolution (Image Filtering) | Apply a 2D convolution kernel to a single-channel image with configurable zero-padding and stride. | https://www.tensortonic.com/problems/conv2d-image-filtering |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Cosine Embedding Loss | Compute cosine embedding loss for similar and dissimilar vector pairs using labels and a configurable margin. | https://www.tensortonic.com/problems/cosine-embedding-loss |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Cumulative Returns | Convert a sequence of periodic returns into cumulative compounded returns at every time-series position. | https://www.tensortonic.com/problems/cumulative-returns |
| Cyclic Encoding | Encode periodic numeric features as sine and cosine coordinates using a specified cycle length. | https://www.tensortonic.com/problems/cyclic-encoding |
| Data Drift Detection | Detect feature drift by computing total variation distance between reference and production histograms. | https://www.tensortonic.com/problems/data-drift-detection |
| Decision Tree Best Split | Find the best decision-tree split by evaluating candidate feature thresholds and selecting the largest impurity reduction. | https://www.tensortonic.com/problems/decision-tree-split |
| Implement Dice Loss | Compute Dice loss for segmentation predictions using overlap, total mass, and a numerical smoothing term. | https://www.tensortonic.com/problems/dice-loss |
| Differencing | Transform a time series into lagged differences while preserving the requested differencing interval. | https://www.tensortonic.com/problems/differencing |
| Discounted Returns | Compute discounted reinforcement-learning returns backward through a reward sequence using a discount factor. | https://www.tensortonic.com/problems/discount-returns |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Double Exponential Smoothing | Apply Holt double exponential smoothing to a time series by updating level and trend components. | https://www.tensortonic.com/problems/double-exponential-smoothing |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Edit Distance | Compute Levenshtein edit distance between two strings using dynamic programming over insertions, deletions, and substitutions. | https://www.tensortonic.com/problems/edit-distance |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| ε-Greedy Action Selection | Select a reinforcement-learning action with epsilon-greedy exploration using action values and controlled randomness. | https://www.tensortonic.com/problems/epsilon-greedy |
| ETL Deduplication | Deduplicate ETL records by configured key fields while applying the required policy for repeated entries. | https://www.tensortonic.com/problems/etl-deduplication |
| ETL Dependency Orchestration | Resolve ETL job dependencies into a valid execution order while detecting missing or cyclic dependencies. | https://www.tensortonic.com/problems/etl-dependency-orchestration |
| ETL Schema Validation | Validate ETL records against required field names and data types, reporting rows that violate the schema. | https://www.tensortonic.com/problems/etl-schema-validation |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Calibration Error | Calculate expected calibration error by binning prediction confidence and weighting accuracy-confidence gaps. | https://www.tensortonic.com/problems/expected-calibration-error |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Exponential Moving Average | Calculate an exponential moving average across a time series using the configured smoothing factor. | https://www.tensortonic.com/problems/exponential-moving-average |
| Feature Store Lookup | Combine stored offline and request-time features in input order, using defaults for unknown user IDs. | https://www.tensortonic.com/problems/feature-store-lookup |
| Implement Focal Loss | Compute mean binary focal loss from predicted probabilities using a configurable focusing parameter. | https://www.tensortonic.com/problems/focal-loss |
| Frequency Encoding | Replace categorical values with their observed frequencies while preserving the original sequence order. | https://www.tensortonic.com/problems/frequency-encoding |
| Generalized Advantage Estimation | Compute generalized advantage estimates backward through rewards, values, discounting, and trace decay. | https://www.tensortonic.com/problems/gae-computation |
| Gaussian Blur Kernel | Generate a normalized 2D Gaussian blur kernel from an odd kernel size and positive standard deviation. | https://www.tensortonic.com/problems/gaussian-blur-kernel |
| Gaussian Naive Bayes | Fit Gaussian Naive Bayes class statistics and predict labels from priors and feature likelihoods. | https://www.tensortonic.com/problems/gaussian-naive-bayes |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Geometric Probability Mass Function & Mean | Compute the geometric distribution probability mass and mean from a valid success probability. | https://www.tensortonic.com/problems/geometric-pmf-mean |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Histogram Equalization | Equalize a grayscale image histogram using its cumulative distribution to remap pixel intensities. | https://www.tensortonic.com/problems/histogram-equalization |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Image Histogram | Count grayscale image pixels into intensity bins and return the histogram in ascending intensity order. | https://www.tensortonic.com/problems/image-histogram |
| Image Rotation (Nearest Neighbor) | Rotate a 2D image around its center with nearest-neighbor sampling and defined out-of-bounds handling. | https://www.tensortonic.com/problems/image-rotation-nearest |
| Impute Missing Values (mean/median) | Impute missing numeric values column-wise with either the mean or median while leaving observed values unchanged. | https://www.tensortonic.com/problems/impute-missing |
| Implement InfoNCE Loss | Compute InfoNCE contrastive loss from query and key embeddings using temperature-scaled similarities. | https://www.tensortonic.com/problems/info-nce-loss |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| Interaction Features | Create pairwise interaction features by multiplying selected input columns while preserving original samples. | https://www.tensortonic.com/problems/interaction-features |
| Intersection over Union (IoU) | Compute intersection over union for two axis-aligned bounding boxes from overlap and combined area. | https://www.tensortonic.com/problems/iou-bounding-box |
| Isotonic Regression Calibration | Calibrate prediction scores with isotonic regression while producing a monotonic non-decreasing mapping. | https://www.tensortonic.com/problems/isotonic-calibration |
| Item-Based CF Prediction | Predict a user-item rating with item-based collaborative filtering from rated neighbors and item similarities. | https://www.tensortonic.com/problems/item-cf-predict |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| K-Means Centroid Update | Update K-means centroids as cluster means while applying the required behavior for empty clusters. | https://www.tensortonic.com/problems/k-means-centroid-update |
| K-Fold Split (Indices Only) | Generate deterministic K-fold train and validation index splits that use every sample exactly once for validation. | https://www.tensortonic.com/problems/kfold-split |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Label Smoothing Loss | Compute multiclass cross-entropy with label smoothing by distributing target mass across all classes. | https://www.tensortonic.com/problems/label-smoothing-loss |
| Lag Features | Generate lagged time-series features for specified offsets with defined padding for unavailable history. | https://www.tensortonic.com/problems/lag-features |
| L-BFGS Two-Loop Recursion | Implement the L-BFGS two-loop recursion to transform a gradient using stored correction-vector history. | https://www.tensortonic.com/problems/lbfgs-two-loop |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Interpolation | Interpolate values between known one-dimensional points while handling exact coordinates and valid boundaries. | https://www.tensortonic.com/problems/linear-interpolation |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Learning Rate Scheduler (Linear Decay) | Compute a linearly decaying learning rate across training steps between configured start and end values. | https://www.tensortonic.com/problems/linear-lr-scheduler |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Log Loss (Per-Sample) | Compute binary log loss for each prediction with clipped probabilities to prevent undefined logarithms. | https://www.tensortonic.com/problems/log-loss-per-sample |
| Log Transform | Apply a numerically safe logarithmic transform to numeric features using the required offset or base. | https://www.tensortonic.com/problems/log-transform |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Factorization SGD Step | Perform one matrix-factorization SGD update for a rated user-item pair with latent factors and regularization. | https://www.tensortonic.com/problems/matrix-factorization-sgd-step |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling 2D | Apply non-overlapping 2D max pooling to rectangular inputs while discarding incomplete edge windows. | https://www.tensortonic.com/problems/max-pooling-2d |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Monte Carlo Policy Evaluation | Estimate state values from complete Monte Carlo episodes by averaging discounted returns for each visited state. | https://www.tensortonic.com/problems/mc-policy-evaluation |
| Compute Mean Average Precision (mAP) | Compute mean average precision across ranked retrieval results from per-query relevance labels. | https://www.tensortonic.com/problems/mean-average-precision |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Rating Imputation | Fill missing user-item ratings with the required row or column mean while preserving observed ratings. | https://www.tensortonic.com/problems/mean-rating-imputation |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Min-Max Scaling | Scale numeric values to a requested range using observed minimum and maximum values with constant-input handling. | https://www.tensortonic.com/problems/min-max-scaling |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Model Versioning | Select a production model by highest accuracy, then lower latency, then the most recent timestamp. | https://www.tensortonic.com/problems/model-versioning-basics |
| Monitoring Metrics Selection | Compute the required monitoring metrics for classification, regression, or ranking prediction results. | https://www.tensortonic.com/problems/monitoring-metrics-selection |
| Morphological Erosion and Dilation | Apply binary-image erosion and dilation with a structuring element and explicit neighborhood boundaries. | https://www.tensortonic.com/problems/morphological-operations |
| Moving Median | Compute a moving median over complete fixed-size sliding windows in an ordered numeric time series. | https://www.tensortonic.com/problems/moving-median |
| Implement Nadam (Nesterov + Adam) | Implement one Nadam optimizer step in NumPy by combining Adam moments with Nesterov momentum. | https://www.tensortonic.com/problems/nadam-optimizer |
| Naive Bayes Log-Likelihood (Bernoulli) | Compute Bernoulli Naive Bayes log-likelihoods from binary features, class priors, and feature probabilities. | https://www.tensortonic.com/problems/naive-bayes-bernoulli |
| NDCG (Normalized Discounted Cumulative Gain) | Calculate normalized discounted cumulative gain at K from ranked relevance scores and their ideal ordering. | https://www.tensortonic.com/problems/ndcg |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| Non-Maximum Suppression | Apply non-maximum suppression to scored bounding boxes using intersection over union and a threshold. | https://www.tensortonic.com/problems/non-maximum-suppression |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Novelty Score | Measure recommendation novelty from item popularity by averaging self-information across recommended items. | https://www.tensortonic.com/problems/novelty-score |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Ordinal Encoding | Map ordered categorical values to integer ranks using a supplied category ordering and preserve input order. | https://www.tensortonic.com/problems/ordinal-encoding |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Percent Change | Compute period-over-period percentage changes in a numeric time series with defined initial-value handling. | https://www.tensortonic.com/problems/percent-change |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| Polynomial Features | Expand numeric inputs into polynomial features through a specified degree using deterministic column ordering. | https://www.tensortonic.com/problems/polynomial-features |
| Popularity Ranking | Rank items by interaction popularity with deterministic ordering for equal-frequency items. | https://www.tensortonic.com/problems/popularity-ranking |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Prioritized Experience Replay | Compute prioritized replay sampling probabilities and normalized importance weights from transition priorities. | https://www.tensortonic.com/problems/priority-replay-sample |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Implement R² Score (Coefficient of Determination) | Compute the coefficient of determination from targets and predictions with explicit constant-target handling. | https://www.tensortonic.com/problems/r2-score |
| Random Forest Majority Vote | Combine multiple decision-tree predictions with majority voting and deterministic handling of tied classes. | https://www.tensortonic.com/problems/random-forest-vote |
| Rank Transform | Replace numeric values with their ranks while applying the specified policy to tied observations. | https://www.tensortonic.com/problems/rank-transform |
| Rating Normalization | Normalize user-item ratings around the required mean while preserving missing-rating markers. | https://www.tensortonic.com/problems/rating-normalization |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| Replay Buffer Sample | Sample a reproducible mini-batch of transitions from a replay buffer without modifying stored experience. | https://www.tensortonic.com/problems/replay-buffer-sample |
| Retraining Trigger Design | Evaluate production monitoring signals and decide whether they satisfy the configured model-retraining policy. | https://www.tensortonic.com/problems/retraining-trigger-design |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Backward (Vanilla RNN) | Backpropagate through one vanilla RNN timestep to compute input, hidden-state, weight, and bias gradients. | https://www.tensortonic.com/problems/rnn-step-backward |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Robust Scaling | Scale numeric features using their median and interquartile range with constant-spread handling. | https://www.tensortonic.com/problems/robust-scaling |
| Compute ROC Curve from Scores | Construct ROC curve thresholds with corresponding true-positive and false-positive rates from binary scores. | https://www.tensortonic.com/problems/roc-curve |
| ROI Pooling | Pool variable-size regions of interest into fixed spatial output grids using per-bin maximum values. | https://www.tensortonic.com/problems/roi-pooling |
| Rolling Standard Deviation | Compute rolling standard deviation over complete time-series windows using the required variance convention. | https://www.tensortonic.com/problems/rolling-standard-deviation |
| Rotate 3D Point Around Z-Axis | Rotate a 3D point around the z-axis by a given angle while preserving its z coordinate. | https://www.tensortonic.com/problems/rotate-around-z |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| SARSA Update | Perform one on-policy SARSA action-value update from the observed reward and next selected action. | https://www.tensortonic.com/problems/sarsa-update |
| Seasonal Average | Estimate seasonal averages by grouping time-series observations at the same position in each period. | https://www.tensortonic.com/problems/seasonal-average |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Shadow Deployment Evaluation | Compare shadow and production model outcomes using the evaluation criteria defined by the problem. | https://www.tensortonic.com/problems/shadow-deployment-evaluation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Compute Silhouette Score | Compute the mean silhouette score from intra-cluster and nearest-cluster distances for labeled samples. | https://www.tensortonic.com/problems/silhouette-score |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Simple Moving Average | Compute the simple moving average over complete fixed-size windows of a numeric time series. | https://www.tensortonic.com/problems/simple-moving-average |
| Sobel Edge Detection | Detect image edges by applying horizontal and vertical Sobel kernels and combining gradient magnitudes. | https://www.tensortonic.com/problems/sobel-edge-detection |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Stratified Train/Test Split | Split indices into train and test sets while approximately preserving the class distribution of each label. | https://www.tensortonic.com/problems/stratified-split |
| Streaming Min-Max Normalization | Update per-feature running minima and maxima, then normalize each incoming numeric batch with the new state. | https://www.tensortonic.com/problems/streaming-minmax |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Target Encoding | Encode each categorical value with the mean target observed for its category while preserving row order. | https://www.tensortonic.com/problems/target-encoding |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Implement TF-IDF Vectorizer | Build TF-IDF document vectors from token counts and inverse document frequency across a text corpus. | https://www.tensortonic.com/problems/tfidf-vectorizer |
| Top-K Recommendations | Return each user's highest-scoring unseen items with deterministic ranking and a configurable result limit. | https://www.tensortonic.com/problems/top-k-recommendations |
| Detect Train-Serving Skew | Detect train-serving skew by comparing offline and online feature values under configured tolerances. | https://www.tensortonic.com/problems/train-serving-skew |
| Implement Triplet Loss | Compute triplet loss from anchor, positive, and negative embeddings using distances and a margin. | https://www.tensortonic.com/problems/triplet-loss |
| User-Based CF Prediction | Predict a user-item rating from similar users' ratings with neighborhood weighting and mean adjustment. | https://www.tensortonic.com/problems/user-based-cf-prediction |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| Warmup + Linear Decay LR Schedule | Compute a learning-rate schedule with linear warmup followed by linear decay across training steps. | https://www.tensortonic.com/problems/warmup-decay-lr |
| Implement Wasserstein Critic Loss | Compute Wasserstein critic loss as the difference between mean fake and real critic scores. | https://www.tensortonic.com/problems/wasserstein-critic-loss |
| Weighted Moving Average | Compute a weighted moving average over complete time-series windows using normalized supplied weights. | https://www.tensortonic.com/problems/weighted-moving-average |
| Winsorization | Winsorize numeric data by clipping values below and above the requested percentile thresholds. | https://www.tensortonic.com/problems/winsorization |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| Block Attention Residuals | Implement Kimi K3 block attention residuals with block-local history, learned mixing weights, and normalized outputs. | https://www.tensortonic.com/research/kimik3/k3-block-attnres |
| Full Attention Residuals | Implement Kimi K3 full attention residuals by mixing normalized layer outputs with learned residual coefficients. | https://www.tensortonic.com/research/kimik3/k3-full-attnres |
| Gated MLA | Build Kimi K3 Gated MLA with low-rank query and KV projections, per-head gating, RoPE inputs, and latent caching. | https://www.tensortonic.com/research/kimik3/k3-gated-mla |
| KDA Recurrence | Implement Kimi K3's KDA recurrence with gated state updates, normalized queries and keys, and ordered token readouts. | https://www.tensortonic.com/research/kimik3/k3-kda-recurrence |
| Quantile Balancing | Implement Kimi K3 quantile balancing by ranking expert loads and computing bounded routing-bias updates. | https://www.tensortonic.com/research/kimik3/k3-quantile-balancing |
| SiTU-GLU | Implement Kimi K3 SiTU-GLU by combining sigmoid-tanh gated activations with an elementwise linear branch. | https://www.tensortonic.com/research/kimik3/k3-situ-glu |
| Stable LatentMoE | Build Kimi K3 Stable LatentMoE routing with latent tokens, expert selection, weighted dispatch, and output reconstruction. | https://www.tensortonic.com/research/kimik3/k3-stable-latent-moe |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/3393979033)
<!-- tensortonic:end -->
