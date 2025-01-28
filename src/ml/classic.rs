pub mod decision_tree;
pub mod gradient_boost;
pub mod k_means;
pub mod k_nearest;
pub mod linear_regression;
pub mod logistic_regression;
pub mod naive_bayes;
pub mod random_forest;
pub mod svm;
pub mod xgboost;

// Re-export public types and functions
pub use decision_tree::{DecisionTree, DecisionTreeAlgorithm, DecisionTreeNode};
pub use k_means::{kmeans, KMeansConfig};
pub use k_nearest::KNNClassifier;
pub use linear_regression::LinearRegression;
pub use logistic_regression::LogisticRegression;
pub use random_forest::RandomForest;
pub use svm::{Kernel, LinearKernel, PolynomialKernel, RBFKernel, SVMConfig, SVM};
pub use xgboost::{XGBConfig, XGBObjective, XGBoostModel};
