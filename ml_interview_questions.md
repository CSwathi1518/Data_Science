# Machine Learning Interview Questions and Answers

## 1. Linear Regression

### Q1: What is linear regression and what are its assumptions?
**Answer:** Linear regression is a supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables using a linear equation. The key assumptions are:
- Linearity: The relationship between X and Y is linear
- Independence: Observations are independent of each other
- Homoscedasticity: Constant variance of residuals
- Normality: Residuals are normally distributed
- No multicollinearity: Independent variables are not highly correlated

### Q2: What are the key metrics used to evaluate linear regression models?
**Answer:**
- **R-squared (R²)**: Proportion of variance in dependent variable explained by independent variables (0 to 1, higher is better)
- **Adjusted R-squared**: R² adjusted for number of predictors, penalizes unnecessary variables
- **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target variable
- **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values

### Q3: How do you handle multicollinearity in linear regression?
**Answer:** Multicollinearity occurs when independent variables are highly correlated. Solutions include:
- Calculate Variance Inflation Factor (VIF) to detect multicollinearity (VIF > 10 indicates high multicollinearity)
- Remove one of the correlated variables
- Combine correlated variables using PCA or feature engineering
- Use regularization techniques (Ridge or Lasso regression)
- Collect more data to reduce correlation

## 2. Logistic Regression

### Q1: What is logistic regression and when is it used?
**Answer:** Logistic regression is a supervised learning algorithm used for binary and multiclass classification problems. It predicts the probability of an instance belonging to a particular class using the sigmoid/logistic function. The output is bounded between 0 and 1. It's used when the target variable is categorical, such as spam detection, disease diagnosis, or customer churn prediction.

### Q2: What are the key evaluation metrics for logistic regression?
**Answer:**
- **Accuracy**: Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **Precision**: TP/(TP+FP) - proportion of positive predictions that are correct
- **Recall/Sensitivity**: TP/(TP+FN) - proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall: 2*(Precision*Recall)/(Precision+Recall)
- **ROC-AUC**: Area under ROC curve, measures model's ability to distinguish between classes
- **Log Loss**: Measures uncertainty of predictions based on divergence from actual labels

### Q3: What is the difference between linear and logistic regression?
**Answer:**
- **Output**: Linear regression predicts continuous values; logistic regression predicts probabilities for categorical outcomes
- **Function**: Linear uses identity function; logistic uses sigmoid function
- **Loss Function**: Linear uses MSE; logistic uses log loss (cross-entropy)
- **Range**: Linear output is unbounded; logistic output is between 0 and 1
- **Use Case**: Linear for regression problems; logistic for classification problems

## 3. Decision Tree, Random Forest, and Bagging

### Q1: How does a decision tree work and what are its advantages and disadvantages?
**Answer:** A decision tree splits data into subsets based on feature values, creating a tree structure with decision nodes and leaf nodes. Splitting criteria include Gini impurity, entropy, or information gain.

**Advantages:**
- Easy to understand and interpret
- Requires little data preprocessing
- Handles both numerical and categorical data
- Can capture non-linear relationships

**Disadvantages:**
- Prone to overfitting
- Unstable - small changes in data can result in different trees
- Biased toward features with more levels
- Cannot extrapolate beyond training data range

### Q2: What is Random Forest and how does it improve upon decision trees?
**Answer:** Random Forest is an ensemble learning method that creates multiple decision trees during training and outputs the mode (classification) or mean (regression) of individual trees. It improves upon decision trees through:
- **Bootstrap Aggregating (Bagging)**: Each tree is trained on a random sample with replacement
- **Feature Randomness**: At each split, only a random subset of features is considered
- **Reduced Overfitting**: Averaging multiple trees reduces variance
- **Better Generalization**: More robust to outliers and noise
- **Feature Importance**: Can measure feature importance across all trees

### Q3: Explain bagging and how it reduces model variance.
**Answer:** Bagging (Bootstrap Aggregating) is an ensemble technique that:
1. Creates multiple subsets of training data through bootstrap sampling (random sampling with replacement)
2. Trains a separate model on each subset
3. Combines predictions through voting (classification) or averaging (regression)

**Variance Reduction:** By training on different subsets and averaging predictions, bagging reduces the impact of outliers and noise in any single training set. The variance of the ensemble is approximately σ²/n where σ² is variance of individual models and n is number of models. This makes the model more stable and less prone to overfitting.

## 4. KNN, AdaBoost, and Gradient Boost

### Q1: How does K-Nearest Neighbors (KNN) algorithm work and what are its key hyperparameters?
**Answer:** KNN is a lazy learning algorithm that classifies new data points based on the majority class of K nearest neighbors in the feature space.

**Process:**
1. Choose number of neighbors K
2. Calculate distance between query point and all training samples
3. Select K nearest neighbors
4. For classification: assign majority class; for regression: assign average value

**Key Hyperparameters:**
- **K**: Number of neighbors (odd number for binary classification to avoid ties)
- **Distance Metric**: Euclidean, Manhattan, Minkowski, or Hamming
- **Weight Function**: Uniform (all neighbors equal) or distance-based (closer neighbors have more weight)

### Q2: What is AdaBoost and how does it work?
**Answer:** AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple weak learners (typically decision stumps) to create a strong learner.

**Process:**
1. Initialize equal weights for all training samples
2. Train a weak learner on weighted training data
3. Calculate error rate of the weak learner
4. Increase weights of misclassified samples
5. Calculate learner's weight based on its accuracy
6. Repeat for specified number of iterations
7. Final prediction is weighted majority vote

**Key Characteristics:**
- Focuses on hard-to-classify examples by increasing their weights
- Sequential training (each model depends on previous ones)
- Sensitive to noisy data and outliers

### Q3: How does Gradient Boosting differ from AdaBoost?
**Answer:**
- **Optimization Approach**: AdaBoost adjusts sample weights; Gradient Boosting fits new models to residual errors (pseudo-residuals)
- **Loss Function**: AdaBoost uses exponential loss; Gradient Boosting can use various differentiable loss functions (MSE, log loss, etc.)
- **Flexibility**: Gradient Boosting is more flexible and can be adapted to different loss functions
- **Error Correction**: AdaBoost focuses on misclassified samples; Gradient Boosting fits gradients of loss function
- **Robustness**: Gradient Boosting is generally more robust to outliers
- **Applications**: Gradient Boosting works for both classification and regression; AdaBoost primarily for classification

## 5. K-Means, Hierarchical, and DBSCAN Clustering

### Q1: Explain the K-Means clustering algorithm and its limitations.
**Answer:** K-Means is an unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster sum of squares.

**Algorithm:**
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

**Limitations:**
- Must specify K beforehand
- Sensitive to initial centroid placement
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Cannot handle non-convex cluster shapes
- Works only with numerical data

### Q2: What is hierarchical clustering and what are its two main approaches?
**Answer:** Hierarchical clustering creates a tree-like structure (dendrogram) of nested clusters without requiring predefined number of clusters.

**Two Approaches:**

**Agglomerative (Bottom-up):**
- Starts with each point as a separate cluster
- Iteratively merges closest clusters
- Continues until single cluster remains
- Linkage criteria: single, complete, average, Ward's method

**Divisive (Top-down):**
- Starts with all points in one cluster
- Iteratively splits clusters
- Continues until each point is separate cluster

**Advantages:** Don't need to specify K, provides hierarchical structure, deterministic results

**Disadvantages:** Computationally expensive O(n³), sensitive to noise and outliers, cannot undo merges/splits

### Q3: How does DBSCAN work and what are its advantages over K-Means?
**Answer:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together.

**Algorithm:**
- **Core Points**: Points with at least minPts neighbors within epsilon radius
- **Border Points**: Non-core points within epsilon of a core point
- **Noise Points**: Points that are neither core nor border points

**Advantages over K-Means:**
- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers (labels them as noise)
- Can identify clusters of varying densities
- Deterministic results

**Parameters:** Epsilon (radius) and minPts (minimum points to form dense region)

## 6. PCA, Variance/Bias, L1, L2, and Elastic Net

### Q1: What is Principal Component Analysis (PCA) and when should it be used?
**Answer:** PCA is an unsupervised dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance lies on the first coordinate (principal component), second greatest on second coordinate, etc.

**Process:**
1. Standardize the data
2. Calculate covariance matrix
3. Compute eigenvectors and eigenvalues
4. Sort eigenvectors by eigenvalues in descending order
5. Select top k eigenvectors
6. Transform original data

**Use Cases:**
- Reduce dimensionality while retaining most variance
- Remove multicollinearity
- Visualize high-dimensional data
- Speed up training
- Reduce storage requirements

**Note:** PCA assumes linear relationships and may lose interpretability of original features.

### Q2: Explain the bias-variance tradeoff in machine learning.
**Answer:** The bias-variance tradeoff is the balance between two sources of error in predictive models.

**Bias:**
- Error from incorrect assumptions in learning algorithm
- High bias leads to underfitting
- Model is too simple, misses relevant relations
- Example: Using linear model for non-linear data

**Variance:**
- Error from sensitivity to fluctuations in training data
- High variance leads to overfitting
- Model learns noise and random fluctuations
- Example: Deep decision tree memorizing training data

**Tradeoff:**
- Total Error = Bias² + Variance + Irreducible Error
- Increasing model complexity: decreases bias, increases variance
- Goal: Find optimal complexity that minimizes total error

**Solutions:** Cross-validation, regularization, ensemble methods, proper feature selection

### Q3: Compare L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
**Answer:**

**L1 Regularization (Lasso):**
- Penalty: λ Σ|wi|
- Performs feature selection by shrinking some coefficients to exactly zero
- Produces sparse models
- Useful when many features are irrelevant
- Not differentiable at zero

**L2 Regularization (Ridge):**
- Penalty: λ Σwi²
- Shrinks coefficients toward zero but never exactly to zero
- Handles multicollinearity well
- Distributes weights among correlated features
- Differentiable everywhere

**Elastic Net:**
- Penalty: λ₁ Σ|wi| + λ₂ Σwi²
- Combines L1 and L2 regularization
- Performs feature selection while handling correlated features
- More stable than Lasso when features are correlated
- Two hyperparameters to tune

**When to Use:** Lasso for feature selection, Ridge when all features are relevant, Elastic Net for correlated features with feature selection.

## 7. NLP, Transfer Learning, Embeddings, CBOW, and Skip-gram

### Q1: What are word embeddings and why are they important in NLP?
**Answer:** Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points.

**Importance:**
- Capture semantic relationships and context
- Reduce dimensionality compared to one-hot encoding
- Enable arithmetic operations (king - man + woman = queen)
- Transfer learning: Pre-trained embeddings can be used for various tasks
- Handle vocabulary better than traditional methods

**Popular Methods:**
- Word2Vec (CBOW and Skip-gram)
- GloVe (Global Vectors)
- FastText
- Contextual embeddings (ELMo, BERT)

**Advantages:** Capture similarity, reduce sparsity, generalize better

### Q2: Explain the difference between CBOW and Skip-gram models.
**Answer:**

**CBOW (Continuous Bag of Words):**
- Predicts target word from context words
- Input: Surrounding context words
- Output: Center/target word
- Architecture: Average context word vectors, predict center word
- Faster to train
- Better for frequent words
- Smooths over distributional information

**Skip-gram:**
- Predicts context words from target word
- Input: Center/target word
- Output: Surrounding context words
- Architecture: Use center word to predict each context word
- Slower to train
- Better for rare words and smaller datasets
- Preserves word order information better

**Example:** For sentence "the cat sits on the mat" with window size 2:
- CBOW: [the, cat, on, the] → sits
- Skip-gram: sits → [the, cat, on, the]

### Q3: What is transfer learning in NLP and what are common approaches?
**Answer:** Transfer learning is using knowledge from a pre-trained model on a large corpus to improve performance on a downstream task with limited data.

**Common Approaches:**

**Feature-based:**
- Use pre-trained embeddings (Word2Vec, GloVe) as fixed features
- Train task-specific model on top
- Embeddings remain frozen

**Fine-tuning:**
- Start with pre-trained language model (BERT, GPT, RoBERTa)
- Add task-specific layers
- Fine-tune entire model or last few layers on target task
- Adjusts weights to task-specific data

**Popular Models:**
- BERT: Bidirectional encoder, masked language modeling
- GPT: Unidirectional decoder, autoregressive generation
- T5: Text-to-text framework
- RoBERTa: Optimized BERT training

**Benefits:** Faster training, better performance with less data, captures general language understanding

## 8. Self-Attention and Transformers

### Q1: What is self-attention mechanism and how does it work?
**Answer:** Self-attention allows a model to weigh the importance of different parts of the input sequence when processing each element.

**Process:**
1. Create Query (Q), Key (K), and Value (V) vectors for each input token through linear transformations
2. Calculate attention scores: score(Q, K) = Q · K^T / √dk (scaled dot-product)
3. Apply softmax to get attention weights
4. Compute weighted sum of values: Attention(Q,K,V) = softmax(QK^T/√dk)V

**Key Characteristics:**
- Captures dependencies regardless of distance
- Parallel computation (unlike RNNs)
- Attention weights show which tokens are important
- √dk scaling prevents softmax saturation for large dimensions

**Benefits:** Long-range dependencies, parallelization, interpretability through attention weights

### Q2: Explain the Transformer architecture and its key components.
**Answer:** Transformer is a neural network architecture based entirely on attention mechanisms, without recurrence or convolution.

**Key Components:**

**Encoder:**
- Multi-head self-attention layer
- Position-wise feed-forward network
- Layer normalization and residual connections
- Positional encoding to capture sequence order

**Decoder:**
- Masked multi-head self-attention (prevents attending to future tokens)
- Multi-head cross-attention (attends to encoder output)
- Position-wise feed-forward network
- Layer normalization and residual connections

**Multi-Head Attention:**
- Runs multiple attention mechanisms in parallel
- Each head learns different aspects of relationships
- Outputs are concatenated and linearly transformed

**Advantages:** Parallelization, long-range dependencies, state-of-the-art performance, foundation for BERT/GPT

### Q3: What is the purpose of positional encoding in Transformers?
**Answer:** Since Transformers process all tokens in parallel without recurrence or convolution, they have no inherent notion of token order or position in sequence.

**Purpose of Positional Encoding:**
- Inject information about relative or absolute position of tokens
- Enable model to use sequence order
- Essential for understanding temporal relationships

**Implementation:**
- Added to input embeddings before first layer
- Uses sine and cosine functions of different frequencies:
  - PE(pos, 2i) = sin(pos / 10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

**Properties:**
- Deterministic (not learned)
- Unique encoding for each position
- Can handle sequences longer than training examples
- Relative positions can be represented by linear transformations

**Alternatives:** Learned positional embeddings, relative positional encodings

## 9. RNN, LSTM, and GRU

### Q1: What are Recurrent Neural Networks (RNN) and what problems do they solve?
**Answer:** RNNs are neural networks designed for sequential data, where the output depends on previous computations through a hidden state.

**Architecture:**
- Hidden state: h_t = f(W_hh * h_(t-1) + W_xh * x_t + b)
- Output: y_t = W_hy * h_t + b
- Same weights shared across all time steps

**Use Cases:**
- Time series prediction
- Natural language processing
- Speech recognition
- Video analysis

**Limitations:**
- Vanishing/exploding gradient problem
- Difficult to capture long-term dependencies
- Sequential processing (not parallelizable)
- Limited memory of past information

### Q2: How do LSTM networks address the limitations of vanilla RNNs?
**Answer:** LSTM (Long Short-Term Memory) uses a gating mechanism to control information flow and maintain long-term dependencies.

**Key Components:**

**Cell State (C_t):**
- Acts as memory highway
- Information flows with minimal modifications

**Three Gates:**

**Forget Gate:**
- f_t = σ(W_f · [h_(t-1), x_t] + b_f)
- Decides what to forget from cell state

**Input Gate:**
- i_t = σ(W_i · [h_(t-1), x_t] + b_i)
- C̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)
- Decides what new information to add

**Output Gate:**
- o_t = σ(W_o · [h_(t-1), x_t] + b_o)
- h_t = o_t * tanh(C_t)
- Decides what to output

**Advantages:** Captures long-term dependencies, mitigates vanishing gradient, selective memory

### Q3: What is GRU and how does it differ from LSTM?
**Answer:** GRU (Gated Recurrent Unit) is a simplified variant of LSTM with fewer parameters and gates.

**Architecture:**

**Reset Gate:**
- r_t = σ(W_r · [h_(t-1), x_t])
- Determines how much past information to forget

**Update Gate:**
- z_t = σ(W_z · [h_(t-1), x_t])
- Controls how much past information to keep

**Candidate Hidden State:**
- h̃_t = tanh(W · [r_t * h_(t-1), x_t])

**Final Hidden State:**
- h_t = (1 - z_t) * h_(t-1) + z_t * h̃_t

**Differences from LSTM:**
- Fewer parameters (2 gates vs 3 gates)
- No separate cell state
- Faster to train
- Often performs similarly to LSTM
- Better for smaller datasets
- LSTM may perform better on tasks requiring precise long-term memory

## 10. Bidirectional and Stacking LSTM

### Q1: What is a Bidirectional LSTM and when should it be used?
**Answer:** Bidirectional LSTM processes sequences in both forward and backward directions, capturing context from both past and future.

**Architecture:**
- Two separate LSTM layers: one processes sequence left-to-right, other right-to-left
- Outputs are concatenated at each time step
- Final output: h_t = [h_forward_t ; h_backward_t]

**Use Cases:**
- Sequence classification (sentiment analysis)
- Named entity recognition
- Machine translation
- Any task where entire sequence is available upfront

**Advantages:**
- Captures full context (past and future)
- Better understanding of ambiguous words
- Improved performance on many NLP tasks

**Limitations:**
- Cannot be used for real-time/online prediction
- Doubled parameters and computation
- Requires entire sequence before processing

### Q2: Explain stacked LSTM and its benefits.
**Answer:** Stacked LSTM (Deep LSTM) consists of multiple LSTM layers stacked on top of each other, where output of one layer becomes input to the next.

**Architecture:**
- Layer 1: Processes raw input sequence
- Layer 2: Processes hidden states from Layer 1
- Layer n: Processes hidden states from Layer n-1
- Can stack many layers (typically 2-4)

**Benefits:**
- **Hierarchical Representation:** Lower layers capture low-level patterns, higher layers capture high-level abstractions
- **Increased Model Capacity:** More parameters to learn complex patterns
- **Better Performance:** Often improves accuracy on complex tasks
- **Feature Hierarchy:** Automatically learns multi-level features

**Best Practices:**
- Use dropout between layers to prevent overfitting
- Start with 2-3 layers, add more if needed
- Monitor for overfitting
- May require more training data

### Q3: What are common techniques to prevent overfitting in LSTM networks?
**Answer:**

**Dropout:**
- Apply dropout to input, output, or recurrent connections
- Typically 0.2-0.5 dropout rate
- Prevents co-adaptation of neurons

**Recurrent Dropout:**
- Dropout applied to recurrent connections
- Same mask used across time steps
- Regularizes hidden state transitions

**Weight Regularization:**
- L1/L2 penalties on weights
- Constrains weight magnitudes
- Prevents extreme weight values

**Early Stopping:**
- Monitor validation loss
- Stop training when validation performance degrades
- Prevents learning noise in training data

**Reduce Network Complexity:**
- Fewer layers or units
- Simpler architecture
- Match complexity to data size

**Data Augmentation:**
- Increase training data size
- Add noise or perturbations
- Back-translation for text

**Batch Normalization/Layer Normalization:**
- Normalize activations
- Stabilizes training
- Regularization effect

## 11. Gradient Descent, Perceptron, ANN, and Activation Functions

### Q1: Explain different variants of gradient descent and their tradeoffs.
**Answer:**

**Batch Gradient Descent:**
- Uses entire dataset to compute gradient
- Update: θ = θ - α ∇J(θ)
- Pros: Stable convergence, guaranteed convergence for convex functions
- Cons: Slow for large datasets, high memory usage, stuck in local minima

**Stochastic Gradient Descent (SGD):**
- Uses single sample to compute gradient
- Updates after each sample
- Pros: Fast, can escape local minima, online learning
- Cons: Noisy updates, fluctuating loss, requires learning rate tuning

**Mini-batch Gradient Descent:**
- Uses small batch (32-512 samples) to compute gradient
- Balance between batch and SGD
- Pros: Efficient computation, stable convergence, leverages vectorization
- Cons: Additional hyperparameter (batch size)

**Advanced Optimizers:**
- **Momentum:** Accelerates SGD by adding fraction of previous update
- **Adam:** Adaptive learning rates per parameter, combines momentum and RMSprop
- **RMSprop:** Adaptive learning rates based on moving average of squared gradients
- **AdaGrad:** Adapts learning rate based on historical gradients

### Q2: What is a perceptron and what are its limitations?
**Answer:** Perceptron is the simplest artificial neural network, consisting of a single neuron with binary threshold activation.

**Architecture:**
- Input: x₁, x₂, ..., xₙ
- Weights: w₁, w₂, ..., wₙ
- Bias: b
- Output: y = 1 if (Σ wᵢxᵢ + b) ≥ 0, else 0

**Learning:**
- Initialize weights randomly
- For each training example, update weights: wᵢ = wᵢ + α(y_true - y_pred)xᵢ
- Repeat until convergence

**Limitations:**
- Can only learn linearly separable patterns
- Cannot solve XOR problem
- Single layer cannot capture complex relationships
- Binary output only
- No hidden layers for feature learning

**Solution:** Multi-layer perceptrons (MLPs) with non-linear activation functions overcome these limitations.

### Q3: Compare different activation functions and their use cases.
**Answer:**

**Sigmoid:**
- Formula: σ(x) = 1 / (1 + e^(-x))
- Range: (0, 1)
- Use: Binary classification output layer
- Pros: Smooth gradient, probabilistic interpretation
- Cons: Vanishing gradient, not zero-centered, computationally expensive

**Tanh:**
- Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Range: (-1, 1)
- Use: Hidden layers in RNNs
- Pros: Zero-centered, stronger gradients than sigmoid
- Cons: Vanishing gradient problem

**ReLU (Rectified Linear Unit):**
- Formula: f(x) = max(0, x)
- Range: [0, ∞)
- Use: Hidden layers in CNNs and deep networks
- Pros: No vanishing gradient, computationally efficient, sparse activation
- Cons: Dying ReLU problem (neurons can become inactive)

**Leaky ReLU:**
- Formula: f(x) = max(αx, x) where α is small (0.01)
- Range: (-∞, ∞)
- Pros: Solves dying ReLU problem
- Cons: Additional hyperparameter

**Softmax:**
- Formula: σ(x)ᵢ = e^(xᵢ) / Σ e^(xⱼ)
- Range: (0, 1), sum to 1
- Use: Multi-class classification output layer
- Pros: Probabilistic interpretation, differentiable

**GELU (Gaussian Error Linear Unit):**
- Smooth approximation of ReLU
- Use: Transformers (BERT, GPT)
- Pros: Better performance in deep networks

## 12. Large Language Models (LLM)

### Q1: What are Large Language Models and what makes them different from traditional NLP models?
**Answer:** LLMs are neural networks with billions of parameters trained on massive text corpora to understand and generate human language.

**Key Characteristics:**
- Scale: Billions to trillions of parameters (GPT-3: 175B, GPT-4: >1T)
- Architecture: Based on Transformer architecture
- Pre-training: Trained on diverse internet-scale data
- Emergent abilities: Capabilities that appear only at large scale
- Few-shot learning: Can perform tasks with minimal examples

**Differences from Traditional Models:**
- **Scale:** Orders of magnitude larger
- **Training Data:** Trained on entire internet vs. task-specific datasets
- **Transfer Learning:** General purpose vs. task-specific
- **Capabilities:** Can perform multiple tasks without retraining
- **Emergent Behaviors:** Reasoning, code generation, mathematical problem-solving
- **In-context Learning:** Learn from prompts without parameter updates

**Examples:** GPT-4, Claude, PaLM, LLaMA, BERT (encoder-only)

### Q2: Explain different training stages of LLMs: pre-training, fine-tuning, and RLHF.
**Answer:**

**Pre-training:**
- Train on massive unlabeled text corpus
- Objective: Predict next token (autoregressive) or masked tokens (masked LM)
- Learns general language understanding, world knowledge, reasoning
- Most computationally expensive phase
- Results in foundation model

**Fine-tuning:**
- Train pre-trained model on specific task or domain
- Supervised learning on labeled examples
- Adjusts model weights for target task
- Types: Full fine-tuning, parameter-efficient (LoRA, adapters)
- Much faster and cheaper than pre-training

**RLHF (Reinforcement Learning from Human Feedback):**
- Aligns model outputs with human preferences
- Steps:
  1. Collect human preferences on model outputs
  2. Train reward model to predict human preferences
  3. Optimize policy using PPO (Proximal Policy Optimization)
- Makes models more helpful, harmless, and honest
- Used in ChatGPT, Claude, and other assistant models

**Instruction Tuning:**
- Fine-tune on diverse instruction-following tasks
- Enables zero-shot task generalization
- Examples: FLAN, InstructGPT

### Q3: What are prompt engineering techniques for improving LLM performance?
**Answer:** Prompt engineering involves crafting inputs to guide LLMs toward desired outputs.

**Key Techniques:**

**Few-shot Learning:**
- Provide examples in the prompt
- Format: Example 1, Example 2, ..., Query
- More examples generally improve performance

**Chain-of-Thought (CoT):**
- Ask model to show reasoning steps
- Prompt: "Let's think step by step"
- Improves performance on reasoning tasks

**Zero-shot CoT:**
- Simply add "Let's think step by step" without examples
- Activates reasoning capabilities

**Role Prompting:**
- Assign a role: "You are an expert Python programmer"
- Primes model with relevant knowledge

**Structured Prompts:**
- Use clear sections: Context, Task, Format, Constraints
- Improves consistency and accuracy

**Self-consistency:**
- Generate multiple responses
- Select most consistent answer
- Reduces errors from randomness

**ReAct (Reasoning + Acting):**
- Interleave reasoning and actions
- Useful for agent-based applications

**System Prompts:**
- Set behavior and constraints
- Define persona, style, limitations

**Best Practices:**
- Be specific and clear
- Provide context
- Specify output format
- Use delimiters for clarity
- Iterate and refine
