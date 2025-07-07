# questions.py
# This file contains all the quiz questions, options, and correct answers

QUIZ_QUESTIONS = {
    "Quiz #1": [
        # Multiple Choice Questions (1-30)
        {
            "question": "Assume arr=np.arange(10). What would be the output of print(arr[2:5])",
            "options": {"a": "2", "b": "[2,3,4]", "c": "[1,2,3]", "d": "[2,3,4,5]"},
            "correct": "b"
        },
        {
            "question": "How do you select the column 'age' from a DataFrame named 'df'?",
            "options": {"a": "df[\"age\"]", "b": "df:age", "c": "df.select('age')", "d": "dr.draw('age')"},
            "correct": "a"
        },
        {
            "question": "Which matplotlib function is used to create multiple subplots?",
            "options": {"a": "plt.figure()", "b": "plt.subplots()", "c": "plt.multi()", "d": "plt.grid()"},
            "correct": "b"
        },
        {
            "question": "Which function in Seaborn is used for creating a heatmap with Seaborn?",
            "options": {"a": "sns.mapplot(data)", "b": "sns.heatgraph(data)", "c": "sns.heatmap(data)",
                        "d": "sns.plot_heat(data)"},
            "correct": "c"
        },
        {
            "question": "Which NumPy function computes the Euclidean distance between vectors A and B?",
            "options": {"a": "np.dot(A,B)", "b": "np.linalg.norm(A - B)", "c": "np.sum(A - B)", "d": "np.mean(A - B)"},
            "correct": "b"
        },
        {
            "question": "For square matrices A and B, which of the trace property is true",
            "options": {"a": "tr(AB)=tr(A)tr(B)", "b": "tr(AB)=tr(BA)", "c": "tr(A)=rank(A)", "d": "tr(A)>0"},
            "correct": "b"
        },
        {
            "question": "If a system of m linear equations with n unknowns is under-determined then",
            "options": {"a": "m=n", "b": "m<n", "c": "m>n", "d": "m=1/n"},
            "correct": "b"
        },
        {
            "question": "What is the sample mode of a set of sample data",
            "options": {"a": "the most frequent value", "b": "the mean value", "c": "the median value",
                        "d": "the largest value"},
            "correct": "a"
        },
        {
            "question": "What does the second quantile encode",
            "options": {"a": "mean", "b": "median", "c": "expected value", "d": "mode"},
            "correct": "b"
        },
        {
            "question": "What characterizes two independent events A and B",
            "options": {"a": "Pr(A|B)=Pr(A)", "b": "Pr(A|B)=Pr(B)", "c": "Pr(A|B)=Pr(A,B)", "d": "Pr(A|B)=Pr(B|A)"},
            "correct": "a"
        },
        {
            "question": "Which of the following encodes the length of a vector",
            "options": {"a": "l2 norm", "b": "l1 norm", "c": "max norm", "d": "frobenius norm"},
            "correct": "a"
        },
        {
            "question": "How can we compare two vectors",
            "options": {"a": "trace", "b": "rank", "c": "inner product", "d": "norm"},
            "correct": "c"
        },
        {
            "question": "Vectors x and y are linear independent if the only solution to ax+by=0 is when",
            "options": {"a": "a=b", "b": "a=-b", "c": "a=b=0", "d": "a,b≠0"},
            "correct": "c"
        },
        {
            "question": "If a matrix has rank 3, what does it mean?",
            "options": {"a": "3 non-zero elements", "b": "3 linear independent rows", "c": "it is 3x3",
                        "d": "it has 3 zero rows"},
            "correct": "b"
        },
        {
            "question": "What is the requirement for matrix inversion",
            "options": {"a": "det(A)=0", "b": "det(A)≠0", "c": "rank(A)=1", "d": "A has a zero row"},
            "correct": "b"
        },
        {
            "question": "Which is the pseudo inverse of A",
            "options": {"a": "A^-1", "b": "A^T*A", "c": "(A^T*A)^-1*A^T", "d": "A*"},
            "correct": "c"
        },
        {
            "question": "What is the objective in least squares regression",
            "options": {"a": "minimizes residuals", "b": "reduces complexity", "c": "maximizes projection",
                        "d": "normalizes the data"},
            "correct": "a"
        },
        {
            "question": "What is the primary goal of gradient descent",
            "options": {"a": "Maximize the loss function", "b": "Minimize the loss function",
                        "c": "Compute the matrix trace", "d": "Find the inverse of a function"},
            "correct": "b"
        },
        {
            "question": "Which function do we minimize in least squares",
            "options": {"a": "convex", "b": "loss", "c": "Entropy", "d": "Likelihood"},
            "correct": "b"
        },
        {
            "question": "What function do we use in logistic regression",
            "options": {"a": "linear", "b": "convex", "c": "sigmoid", "d": "monotonic"},
            "correct": "c"
        },
        {
            "question": "Which of the following is NOT a measure of variability",
            "options": {"a": "range", "b": "variance", "c": "standard deviation", "d": "mean value"},
            "correct": "d"
        },
        {
            "question": "If random variables A and B are independent, then",
            "options": {"a": "P(A,B)=P(A)P(B)", "b": "P(A,B)=P(A)+P(B)", "c": "P(A,B)=P(A|B)", "d": "P(A,B)=P(B|A)"},
            "correct": "a"
        },
        {
            "question": "Which distribution can be used to model multiple coin tosses",
            "options": {"a": "Gaussian", "b": "Uniform", "c": "Binomial", "d": "Poisson"},
            "correct": "c"
        },
        {
            "question": "P(X|Y) is related to P(X,Y) via",
            "options": {"a": "P(X|Y)=P(X,Y)P(Y)", "b": "P(X|Y)=P(X,Y)/P(Y)", "c": "P(X|Y)=P(X,Y)+P(Y)",
                        "d": "P(X|Y)=P(Y|X)"},
            "correct": "b"
        },
        {
            "question": "The joint probability P(X,Y) is equal to",
            "options": {"a": "P(X|Y)", "b": "P(X|Y)P(Y)", "c": "P(X)P(Y)", "d": "P(X)"},
            "correct": "b"
        },
        {
            "question": "In Bayesian models, what does the prior encode",
            "options": {"a": "Observed data", "b": "noise", "c": "Likelihood", "d": "Initial beliefs"},
            "correct": "d"
        },
        {
            "question": "What is the objective in MLE",
            "options": {"a": "Minimize variance of predictions", "b": "Maximize posterior probability",
                        "c": "Maximize likelihood of observed data", "d": "normalize the data"},
            "correct": "c"
        },
        {
            "question": "What is the difference between MAP and MLE",
            "options": {"a": "the consideration of the prior", "b": "calculation of likelihoods",
                        "c": "independent variables", "d": "use of gradients"},
            "correct": "a"
        },
        {
            "question": "In MLE with Gaussian distributions, what loss do we minimize",
            "options": {"a": "MSE", "b": "MAE", "c": "NLL", "d": "DVD"},
            "correct": "c"
        },
        {
            "question": "The cross product between two vectors is",
            "options": {"a": "a matrix", "b": "a vector", "c": "a scalar", "d": "a tensor"},
            "correct": "b"
        },
        # Input Questions (31-36)
        {
            "question": "If P(A)=0.6 and P(B|A)=0.5, then P(A,B) is",
            "type": "input",
            "correct": "0.3"
        },
        {
            "question": "For matrix A=[1,2;3,1], the rank is",
            "type": "input",
            "correct": "2"
        },
        {
            "question": "The gradient of x^3+x+1 at x=2 is",
            "type": "input",
            "correct": "13"
        },
        {
            "question": "If Cov(X,Y)= 10, σx=4 and σy=5, the correlation is",
            "type": "input",
            "correct": "0.5"
        },
        {
            "question": "Assume {1,2,4,5,1}, what is the median and mode",
            "type": "input",
            "correct": "median: 2, mode: 1"
        },
        {
            "question": "Let a=[1,2] and b=[2,3], what is ab",
            "type": "input",
            "correct": "8"
        },
        # True/False Questions (37-41)
        {
            "question": "An invertible matrix is always square",
            "type": "truefalse",
            "correct": "TRUE"
        },
        {
            "question": "Smaller learning rates in gradient descent are better",
            "type": "truefalse",
            "correct": "FALSE"
        },
        {
            "question": "Gradient descent can get stuck in local minima",
            "type": "truefalse",
            "correct": "TRUE"
        },
        {
            "question": "The sample mean is always smaller than the sample mode",
            "type": "truefalse",
            "correct": "FALSE"
        },
        {
            "question": "For events A and B, P(A U B)<= P(A) + P(B)",
            "type": "truefalse",
            "correct": "TRUE"
        }
    ],

    "Quiz #2": [
        # Multiple Choice Questions (1-30)
        {
            "question": "What is the objective of PCA",
            "options": {"a": "Reduce dimensionality while retaining variance", "b": "Increase data complexity",
                        "c": "perform supervised classification", "d": "Maximize reconstruction error"},
            "correct": "a"
        },
        {
            "question": "What is back propagation",
            "options": {"a": "Forward data pass through the network", "b": "Manual update of weights",
                        "c": "Algorithm to compute gradients of the loss", "d": "Random weight initialization"},
            "correct": "c"
        },
        {
            "question": "How does k-means assign points to clusters?",
            "options": {"a": "Based on label frequency", "b": "Using hierarchical distance",
                        "c": "minimizing distance to the nearest centroid", "d": "Random selection of cluster IDs"},
            "correct": "c"
        },
        {
            "question": "A common supervised classification performance metric",
            "options": {"a": "Mean squared error", "b": "Accuracy", "c": "Reconstruction loss",
                        "d": "Euclidean distance"},
            "correct": "b"
        },
        {
            "question": "What type of algorithm is a Random Forest?",
            "options": {"a": "A linear regression model", "b": "A neural network", "c": "An ensemble of decision trees",
                        "d": "A clustering algorithm"},
            "correct": "c"
        },
        {
            "question": "What is the loss of a Variational autoencoder",
            "options": {"a": "reconstruction loss", "b": "reconstruction loss and Kullback-Leibler divergence",
                        "c": "mean squared error", "d": "cross-entropy"},
            "correct": "b"
        },
        {
            "question": "What is the objective of machine learning",
            "options": {"a": "minimize loss", "b": "maximize accurate", "c": "reduce false positives",
                        "d": "maximize cost"},
            "correct": "a"
        },
        {
            "question": "What is the ROC curve used to evaluate?",
            "options": {"a": "Model training time", "b": "Accuracy over time",
                        "c": "Trade-off between true positive and false positive rates",
                        "d": "Distance between clusters"},
            "correct": "c"
        },
        {
            "question": "What is NOT an issue with k-means",
            "options": {"a": "Random initialization", "b": "single cluster assignment",
                        "c": "define number of clusters", "d": "estimate centroids"},
            "correct": "d"
        },
        {
            "question": "What is an appropriate loss function for regression",
            "options": {"a": "accuracy", "b": "mean squared error", "c": "categorical cross entropy", "d": "entropy"},
            "correct": "b"
        },
        {
            "question": "What is the objective of an AutoEncoder",
            "options": {"a": "classify input data", "b": "generate new labeled data",
                        "c": "reconstruct the input from a latent representation", "d": "detect outliers"},
            "correct": "c"
        },
        {
            "question": "What is the goal of the generator in GANs",
            "options": {"a": "Detect fake data", "b": "Create data that fools the discriminator",
                        "c": "Classify real samples", "d": "Minimize reconstruction loss"},
            "correct": "b"
        },
        {
            "question": "An n × n matrix A is diagonalizable if",
            "options": {"a": "all its entries are nonzero", "b": "it has n linearly independent eigenvectors",
                        "c": "it is symmetric", "d": "it has unit norm"},
            "correct": "b"
        },
        {
            "question": "What is the objective of the discriminator in GANs",
            "options": {"a": "generate realistic data samples", "b": "transform noise into structured data",
                        "c": "distinguish between real and generated samples", "d": "minimize the reconstruction loss"},
            "correct": "c"
        },
        {
            "question": "What is the purpose of the kernel trick in SVMs",
            "options": {"a": "reduce the dimensionality of the input", "b": "explicitly compute feature vectors",
                        "c": "transform non-linearly separable data into a higher-dimensional space",
                        "d": "perform stochastic optimization"},
            "correct": "c"
        },
        {
            "question": "What mainly controls the behavior of KNN classifiers",
            "options": {"a": "Learning rate", "b": "Number of hidden layers", "c": "Number of neighbors",
                        "d": "Weight initialization"},
            "correct": "c"
        },
        {
            "question": "What is the classification scenario where examples are annotated with multiple labels",
            "options": {"a": "discrete", "b": "binary", "c": "multi-class", "d": "multi-label"},
            "correct": "d"
        },
        {
            "question": "Which is a symptom of overfitting",
            "options": {"a": "High test accuracy", "b": "Low training loss, high test loss",
                        "c": "High variance in input", "d": "Underutilized GPU"},
            "correct": "b"
        },
        {
            "question": "What is the goal of SVM regarding the separating hyperplane",
            "options": {"a": "minimize the margin", "b": "maximize the margin", "c": "minimize projection error",
                        "d": "reduce dimensionality"},
            "correct": "b"
        },
        {
            "question": "How are non-linearities introduced in ANNs",
            "options": {"a": "activation", "b": "convolution", "c": "inner product", "d": "low rank"},
            "correct": "a"
        },
        {
            "question": "When using ANNs for classification, what is the typical last layer",
            "options": {"a": "softmax", "b": "relu", "c": "convolution", "d": "fully connected"},
            "correct": "a"
        },
        {
            "question": "What is the rank of a matrix",
            "options": {"a": "The largest eigenvalue", "b": "The trace of the matrix",
                        "c": "The number of linearly independent rows or columns", "d": "The sum of all entries"},
            "correct": "c"
        },
        {
            "question": "Which one is NOT an activation function",
            "options": {"a": "sigmoid", "b": "relu", "c": "tanh", "d": "log"},
            "correct": "d"
        },
        {
            "question": "What is updated during backpropagation",
            "options": {"a": "weights", "b": "initialization", "c": "activation", "d": "input"},
            "correct": "a"
        },
        {
            "question": "How do we call the application of a trained ANN",
            "options": {"a": "perceptron", "b": "back propagation", "c": "inference", "d": "activation"},
            "correct": "c"
        },
        {
            "question": "What is supervised learning",
            "options": {"a": "Learning patterns in data without labels", "b": "Optimizing a model",
                        "c": "Learning from labeled data", "d": "Clustering data into groups"},
            "correct": "c"
        },
        {
            "question": "What is the difference between AE and VAE",
            "options": {"a": "VAE uses supervised learning; AE does not", "b": "AE is stochastic; VAE is deterministic",
                        "c": "AE uses latent sampling; VAE does not",
                        "d": "prior distribution on hidden representation"},
            "correct": "d"
        },
        {
            "question": "What controls the behavior of Gradient Descent",
            "options": {"a": "number of output classes", "b": "learning rate", "c": "activation function",
                        "d": "number of nodes"},
            "correct": "b"
        },
        {
            "question": "Key challenge in evaluating the unsupervised learning algorithms",
            "options": {"a": "Absence of labeled ground truth", "b": "Overfitting",
                        "c": "Inability to compute gradients", "d": "Requirement for a validation set"},
            "correct": "a"
        },
        {
            "question": "What is the objective of gradient descent",
            "options": {"a": "maximize the number of features", "b": "compute the eigenvalues",
                        "c": "minimize a loss function", "d": "evaluate model performance"},
            "correct": "c"
        },
        # Input Questions (31-36)
        {
            "question": "Consider a simple AutoEncoder with encoding weights W=[1,0,0;0,1,0] and decoder W^T. For input X=[2;4;6], what is the output?",
            "type": "input",
            "correct": "[2;4;0]"
        },
        {
            "question": "What is the standardized value (z) of x=0.2, if μ=0.1 and σ=10",
            "type": "input",
            "correct": "0.01"
        },
        {
            "question": "Assume the SVD decomposition of X, where U=[1,0;0,1], S=[3,0;0,1] and V=[2,1;0,1]. Calculate X",
            "type": "input",
            "correct": "[6,0;1,1]"
        },
        {
            "question": "What is the output of a perceptron where input is X=[1,2], W=[-4,1;2,1], b=[1;1] and ReLU activation",
            "type": "input",
            "correct": "[0;5]"
        },
        {
            "question": "Assume a current weight is 0.3, the learning rate=0.1 and the gradient is 0.1. What is the new weight after one step of gradient descent",
            "type": "input",
            "correct": "0.29"
        },
        {
            "question": "The eigen-decomposition of matrix A=[2,3;3,2] is associated with eigenvectors v1=[1;1] and v2=[1;-1]. What are the elements of the diagonal matrix?",
            "type": "input",
            "correct": "5 and -1"
        },
        # True/False Questions (37-41)
        {
            "question": "A perceptron with linear activation function can model non-linear decision boundaries",
            "type": "truefalse",
            "correct": "FALSE"
        },
        {
            "question": "Gradient Boosting builds trees sequentially, each correcting the errors of the previous one",
            "type": "truefalse",
            "correct": "TRUE"
        },
        {
            "question": "Random Forest is an ensemble method based on multiple neural networks",
            "type": "truefalse",
            "correct": "FALSE"
        },
        {
            "question": "AUC (Area Under the Curve) of 0.5 indicates a perfect classifier",
            "type": "truefalse",
            "correct": "FALSE"
        },
        {
            "question": "Principal components in PCA are guaranteed to be orthogonal to each other",
            "type": "truefalse",
            "correct": "TRUE"
        }
    ],

    "Quiz #3": [
        # Multiple Choice Questions (1-20)
        {
            "question": "What is the objective of image segmentation",
            "options": {"a": "convert an image to grayscale", "b": "increase image contrast",
                        "c": "assign a class label to each pixel", "d": "detect motion in a video"},
            "correct": "c"
        },
        {
            "question": "What is the objective of image super-resolution",
            "options": {"a": "divide an image into patches",
                        "b": "reconstruct a high-resolution image from a low-resolution input",
                        "c": "detect objects in an image", "d": "reduce the number of channels"},
            "correct": "b"
        },
        {
            "question": "Why MLPs are not ideal for image analysis",
            "options": {"a": "ignore spatial structure", "b": "use too much convolution", "c": "always require GPUs",
                        "d": "can't be trained with backpropagation"},
            "correct": "a"
        },
        {
            "question": "What is the key operator for capturing spatial dependencies",
            "options": {"a": "activation", "b": "convolution", "c": "Normalization", "d": "Matrix multiplication"},
            "correct": "b"
        },
        {
            "question": "What does a CNN pooling layer do",
            "options": {"a": "Normalizes the image", "b": "Increases the resolution",
                        "c": "Applies a fully connected layer", "d": "Reduces spatial dimensions"},
            "correct": "d"
        },
        {
            "question": "What is the goal of softmax activation",
            "options": {"a": "convert logits into a probability distribution", "b": "normalize pixel values",
                        "c": "compute the loss", "d": "backpropagate gradients"},
            "correct": "a"
        },
        {
            "question": "What is the objective of data augmentation",
            "options": {"a": "reduce training time", "b": "generate ground truth",
                        "c": "artificially increase the training data", "d": "compress the dataset"},
            "correct": "c"
        },
        {
            "question": "What is the output of a segmentation network applied on a HxWx3 Image with C potential classes",
            "options": {"a": "H×W×C", "b": "H×W", "c": "HxWx3", "d": "H+W"},
            "correct": "a"
        },
        {
            "question": "What is the objective of object detection",
            "options": {"a": "Localization + Segmentation", "b": "Localization + Classification",
                        "c": "Classification + Segmentation", "d": "Denoising + Localizations"},
            "correct": "b"
        },
        {
            "question": "What is the limitation of per-frame classification of video",
            "options": {"a": "requires 3D convolution", "b": "cannot be parallelized", "c": "always overfits the data",
                        "d": "ignores temporal dependencies"},
            "correct": "d"
        },
        {
            "question": "Which approach is not applicable to time-series analysis",
            "options": {"a": "MLP", "b": "1D CNN", "c": "Temporal CNN", "d": "Spatial CNN"},
            "correct": "d"
        },
        {
            "question": "What is the key feature of RNNs",
            "options": {"a": "maintain hidden state", "b": "process static inputs", "c": "discard past information",
                        "d": "use attention mechanisms"},
            "correct": "a"
        },
        {
            "question": "What is the difference between RNNs and LSTMs",
            "options": {"a": "RNNs are always faster", "b": "LSTMs do not support backpropagation",
                        "c": "LSTMs use gates to control memory, while RNNs",
                        "d": "RNNs use convolution instead of recurrence"},
            "correct": "c"
        },
        {
            "question": "How do we encode text",
            "options": {"a": "embeddings", "b": "activations", "c": "vocabulary sorting", "d": "matrix products"},
            "correct": "a"
        },
        {
            "question": "What is the main purpose of the attention mechanism",
            "options": {"a": "initialize neural network weights", "b": "generate token embeddings",
                        "c": "reduce model parameters", "d": "focus on the most relevant parts of the input"},
            "correct": "d"
        },
        {
            "question": "What is tokenization in transformers",
            "options": {"a": "Splitting text into smaller units", "b": "Compressing text into images",
                        "c": "Applying pooling on sequences", "d": "Normalizing data to unit variance"},
            "correct": "a"
        },
        {
            "question": "How is the location of a token encoded in transformer",
            "options": {"a": "recurrent connections", "b": "positional encoding", "c": "sorting the tokens",
                        "d": "measuring token length"},
            "correct": "b"
        },
        {
            "question": "What type of neural network is a U-Net?",
            "options": {"a": "classification-only architecture", "b": "network for text summarization",
                        "c": "GAN variant", "d": "convolutional network designed for image segmentation"},
            "correct": "d"
        },
        {
            "question": "What is a limitation of RNNs?",
            "options": {"a": "can't process sequential data", "b": "struggle to capture long-term dependencies",
                        "c": "require large images", "d": "don't support backpropagation"},
            "correct": "b"
        },
        {
            "question": "What does \"one-hot encoding\" mean?",
            "options": {"a": "Representing categories as binary vectors with a single 1", "b": "Compressing text",
                        "c": "Sorting the data", "d": "Tokenizing text"},
            "correct": "a"
        }
    ],

    "Quiz #4": [
        # Multiple Choice Questions - ADVANCED TOPICS (No duplicates from Quiz #3)
        {
            "question": "What is the main advantage of ReLU activation function in CNNs?",
            "options": {"a": "prevents overfitting", "b": "reduces model complexity", "c": "avoids saturated gradients",
                        "d": "normalizes input data"},
            "correct": "c"
        },
        {
            "question": "A video is represented as a 4D tensor with dimensions T x 3 x H x W, where T represents",
            "options": {"a": "the number of channels", "b": "the temporal dimension (number of frames)",
                        "c": "the height dimension", "d": "the batch size"},
            "correct": "b"
        },
        {
            "question": "In late fusion for video classification, what is the main approach?",
            "options": {"a": "Use 3D convolutions throughout the network",
                        "b": "Run 2D CNN on each frame, then pool features and feed to classifier",
                        "c": "Process only the first and last frames", "d": "Apply optical flow before classification"},
            "correct": "b"
        },
        {
            "question": "In the image formation model Y = A(X*) + N, what does A represent?",
            "options": {"a": "the noise component", "b": "the forward operator (e.g., blurring, downsampling)",
                        "c": "the clean signal", "d": "the reconstruction algorithm"},
            "correct": "b"
        },
        {
            "question": "What is the main advantage of 3D CNNs over 2D CNNs for video analysis?",
            "options": {"a": "they require less computational resources",
                        "b": "they can capture temporal dependencies through 3D convolutions",
                        "c": "they work only with grayscale videos", "d": "they eliminate the need for pooling layers"},
            "correct": "b"
        },
        {
            "question": "Which performance metric is commonly used to evaluate image super-resolution quality?",
            "options": {"a": "accuracy", "b": "F1-score", "c": "PSNR (Peak Signal-to-Noise Ratio)",
                        "d": "cross-entropy loss"},
            "correct": "c"
        },
        {
            "question": "In GANs for super-resolution, what is the role of the discriminator?",
            "options": {"a": "generate high-resolution images from low-resolution inputs",
                        "b": "distinguish between real high-resolution and generated super-resolved images",
                        "c": "apply noise to the input images", "d": "compress the image data"},
            "correct": "b"
        },
        {
            "question": "What is the main purpose of gates in LSTM networks?",
            "options": {"a": "initialize neural network weights", "b": "generate token embeddings",
                        "c": "reduce model parameters", "d": "control what information to keep or discard from memory"},
            "correct": "d"
        },
        {
            "question": "In sequence modeling, what does 'many-to-one' refer to?",
            "options": {"a": "video captioning", "b": "machine translation", "c": "sentiment classification",
                        "d": "music generation"},
            "correct": "c"
        },
        {
            "question": "What is the main principle behind Word2Vec?",
            "options": {"a": "words with similar frequency appear together",
                        "b": "words that appear in similar contexts have similar meanings",
                        "c": "longer words have better representations",
                        "d": "alphabetical order determines similarity"},
            "correct": "b"
        },
        {
            "question": "What is the main advantage of LSTM over vanilla RNNs?",
            "options": {"a": "faster training time", "b": "smaller memory requirements",
                        "c": "better at preserving information over many timesteps", "d": "simpler architecture"},
            "correct": "c"
        },
        {
            "question": "What causes the vanishing gradient problem in RNNs?",
            "options": {"a": "too many parameters", "b": "gradients become smaller as they backpropagate through time",
                        "c": "insufficient training data", "d": "incorrect learning rate"},
            "correct": "b"
        },
        {
            "question": "In Neural Machine Translation, what is the role of the encoder?",
            "options": {"a": "generates the target sentence", "b": "produces an encoding of the source sentence",
                        "c": "applies attention weights", "d": "tokenizes the input"},
            "correct": "b"
        },
        {
            "question": "What is the key advantage of multi-head attention over single-head attention?",
            "options": {"a": "Uses less computational resources", "b": "Processes only static inputs",
                        "c": "Allows the model to focus on different types of relationships simultaneously",
                        "d": "Eliminates the need for positional encoding"},
            "correct": "c"
        },
        {
            "question": "What problem does Retrieval-Augmented Generation (RAG) primarily solve?",
            "options": {"a": "Slow inference speed", "b": "High computational costs",
                        "c": "Limited knowledge cutoff dates and hallucinations", "d": "Complex model architectures"},
            "correct": "c"
        },
        {
            "question": "In reinforcement learning, what is a policy?",
            "options": {"a": "The reward function", "b": "The function that maps states to actions",
                        "c": "The environment's response", "d": "The discount factor"},
            "correct": "b"
        },
        {
            "question": "What is the main goal of Reinforcement Learning from Human Feedback (RLHF)?",
            "options": {"a": "Reduce training time", "b": "Increase model size",
                        "c": "Align model behavior with human preferences", "d": "Eliminate the need for labeled data"},
            "correct": "c"
        },
        {
            "question": "What is the main advantage of transformer architecture over RNNs?",
            "options": {"a": "Lower computational complexity", "b": "Parallel processing capability",
                        "c": "Simpler architecture", "d": "Better for small datasets"},
            "correct": "b"
        },
        {
            "question": "In reinforcement learning, what is the Q-function?",
            "options": {"a": "Quality of the policy", "b": "Quantum state function", "c": "Action-value function",
                        "d": "Query mechanism"},
            "correct": "c"
        },
        {
            "question": "What is the primary innovation of GPT models?",
            "options": {"a": "Bidirectional training", "b": "Autoregressive language modeling",
                        "c": "Convolutional layers", "d": "Recurrent connections"},
            "correct": "b"
        },
        {
            "question": "What does BERT stand for?",
            "options": {"a": "Bidirectional Encoder Representations from Transformers",
                        "b": "Basic Encoder for Retrieval Tasks", "c": "Binary Encoding with Recurrent Transformers",
                        "d": "Bayesian Estimation and Regression Technique"},
            "correct": "a"
        },
        {
            "question": "In reinforcement learning, what is the exploration-exploitation tradeoff?",
            "options": {"a": "Balancing model complexity and performance",
                        "b": "Balancing trying new actions vs. using known good actions",
                        "c": "Balancing training time and accuracy", "d": "Balancing memory usage and speed"},
            "correct": "b"
        }
    ]
}