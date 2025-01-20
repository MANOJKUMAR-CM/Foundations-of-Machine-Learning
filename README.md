# Foundations of ML

The **Foundations of ML** repo! This is where I’ve tackled some cool ML assignments from my course. Dive in and explore the solutions, experiments, and insights.

## What's Inside?

### Assignment 1: Least Squares and Beyond
- **Task**: Solve the regression problem using both analytical and gradient methods.
- **Highlights**:
  - **Analytical Solution**: Nailed down the least squares solution with good old linear algebra.
  - **Gradient Descent (GD)**: Iteratively updated weights to minimize the error, and plotted how close we got to the analytical solution.
  - **Stochastic Gradient Descent (SGD)**: Used a batch size of 100 for a more dynamic approach, with observations on convergence.
  - **Ridge Regression**: Cross-validated different λ values to find the sweet spot, then compared the test errors of ridge vs. least squares. Spoiler: Ridge often wins when dealing with overfitting.
  - **Kernel Regression**: Explored which kernel works best and justified the choice. Implemented it and tested the predictions.

### Assignment 2: Spam Classifier from Scratch
- **Task**: Build a spam classifier from scratch, no shortcuts.
- **Highlights**:
  - **Preprocessing**: Cleaned up the text data by removing noise, tokenizing, and extracting features.
  - **Logistic Regression**: Implemented the classifier manually, training it to distinguish spam from ham.
  - **Evaluation**: Tuned the model, evaluated its performance, and iterated for improvements. Turns out, a well-tuned model makes all the difference!

### Assignment 3: PCA and K-Means Fun
- **Task 1**: PCA on the MNIST dataset.
  - **Principal Components**: Visualized the major components that capture the essence of the dataset.
  - **Dimensionality Reduction**: Reconstructed images with fewer dimensions, analyzed how much variance each principal component explained, and determined the optimal dimension for accurate classification.
- **Task 2**: K-Means on a 2D dataset.
  - **Lloyd’s Algorithm**: Ran K-means with multiple initializations and plotted error reduction over iterations.
  - **Voronoi Diagrams**: Visualized clusters for different K values using Voronoi regions, showing how data points were grouped.
  - **Clustering Effectiveness**: Critically analyzed whether Lloyd’s algorithm was the best choice and explored alternative methods for clustering.

Happy coding! 🚀

