#ifndef MACHLIB_HPP
#define MACHLIB_HPP

#include <iostream>

#include "Eigen/Dense"	// Defines all member functions for MatrixXd and related types. Everything in this header (and others) is in Eigen namespace.

//using Eigen::Matrix;
//using Eigen::Vector;
//using Eigen::RowVector;
//using Eigen::Dynamic;

using namespace Eigen;

/*
	Conventions:
	Vectors are vertical by default.
	Dataset is examples x features.

	Terms:
	Variables (Features), parameters, result.
	Models (h(x)): Linear regression (pol. reg.), Logistic regression.
	Cost function: Square error.
	Optimization algorithms: Normal equation (lin. reg.), Batch gradient descent, Conjugate gradient, BFGS, L-BFGS, etc.
	Learning algorithm (cost function + optimization algorithm)
	> Normalization: Feature scaling, Mean normalization, Learning rate, Features edition.
	> Overfitting: Use Regularization: delete features and reduce parameters (reg. param.).
	Supervised learning algorithms: Random Forest, Support Vector Machines (SVM), Gradient Boosting Machines (GBM)...
	Optimization algorithms: Linear Programming, Integer Programming, Genetic Algorithms, Reinforcement Learning...
	Convolutional Neural Networks (CNNs)
	sentiment analysis, named entity recognition, topic modeling, and text classification using methods like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, or, of course, the Transformer models
	statistical methods like Isolation Forest, One-Class SVM, or Gaussian Mixture Models (GMM), as well as deep learning approaches such as Autoencoders
*/

// Use binaryExpr()
// Use std::move() maybe

// Constants & enums ----------

double e  = 2.71828182845904523536028747135266249;
double pi = 3.14159265358979323846264338327950288;

enum ModelType {
	LinearRegressionAnalytical,
	LinearRegression,			// for Regression problems (continuous values)
	LogisticRegression			// for Classification systems (discrete values)
};


// Declarations ----------

/**
	Store the dataset (examples x features) and solutions (1 x examples). Rows are examples. Columns are features.
	In dataset, the first row should be all 1s (for the independent term of h)
*/
template<typename T>
class Data
{
public:
	Data(Matrix<T, Dynamic, Dynamic>& data, bool normalize);	//!< Data param contains solution + features (in this order).
	
	size_t numExamples()   { return dataset.rows(); };
	size_t numParams() { return dataset.cols(); };
	size_t numFeatures()   { return dataset.cols() - 1; };

	Matrix<T, Dynamic, Dynamic> dataset;	//!< Matrix<T, Examples, Features>
	Vector<T, Dynamic> solutions;			//!< Vector<T, Examples>
	Vector<T, Dynamic> mean;				//!< for Mean Normalization (make mean == 0)
	Vector<T, Dynamic> range;				//!< for Feature Scaling (make range [-1, 1])
};


/**
	Hypothesis & Learning algorithm (cost function + optimization algorithm)
*/
template<typename T>
class Model
{
	ModelType modelType;

	Vector<T, Dynamic> pow_2(T base, Vector<T, Dynamic>& exponents);	// not used

public:
	Model(ModelType modelType, double alpha, Matrix<T, Dynamic, Dynamic>& dataset, bool normalize)
		: parameters(data.numParams()), modelType(modelType), alpha(alpha), data(dataset, normalize) { }

	Data<T> data;
	Vector<T, Dynamic> parameters;
	double alpha;								//!< Learning rate

	T model(Vector<T, Dynamic> features);		//!< Call h(x), where x == features you provide.
	float costFunction();						//!< Compute cost function (square error cost function)
	Vector<T, Dynamic> optimization();			//!< Optimize parameters with a learning algorithm (batch gradient descent or Normal equation) (alternatives: Conjugate gradient, BFGS, L-BFGS...).
};


// Definitions ----------

template<typename T>
Data<T>::Data(Matrix<T, Dynamic, Dynamic>& data, bool normalize)
	: dataset(data.rows(), data.cols()),
	solutions(data.rows()),
	mean(dataset.cols()),
	range(dataset.cols())
{
	// Dataset & Solution
	dataset = std::move(data);
	dataset.col(0).setConstant(1);			// first column is full of 1s (independent variable)

	solutions = std::move(data.block(0, 0, data.rows(), 1));
	
	// Normalization (Mean normalization & Feature scaling)
	if (normalize)
	{
		mean  = dataset.colwise().sum() / numExamples();
		mean(0) = 0;
		range = dataset.colwise().maxCoeff() - dataset.colwise().minCoeff();
		range(0) = 1;
	}
	else
	{
		mean.setConstant(0);
		range.setConstant(1);
	}
	//std::cout << mean << std::endl;
	//std::cout << dataset << std::endl;
	//std::cout << range << std::endl;
	dataset = (dataset.rowwise() - mean.transpose()).array().rowwise() / range.transpose().array();		// x = (x - mean) / range
	//std::cout << dataset << std::endl;
};

template<typename T>
T Model<T>::model(Vector<T, Dynamic> features)	// < Not a reference because this allows to pass inline-created vector.
{
	// Check dimensions
	if (features.size() != data.numFeatures())
		std::cout << __func__ << " Error: wrong number of features" << std::endl;

	// Add independent variable and normalize
	Vector<T, Dynamic> normFeatures(data.numParams());
	normFeatures.row(0).setConstant(1);
	normFeatures.block(1, 0, features.rows(), 1) = features;
	normFeatures = (normFeatures - data.mean).array() / data.range.array();

	// Apply model
	switch (modelType)
	{
	case LinearRegressionAnalytical:
	case LinearRegression:			// Linear function: h(x)
		return parameters.transpose() * normFeatures;
		break;

	case LogisticRegression:		// Logistic function / Sigmoid: g(x)
		return 1.0 / (1.0 + std::pow((T)e, - parameters.transpose() * normFeatures));
		break;
	
	default:
		// <<< Exception
		break;
	}
}

template<typename T>
float Model<T>::costFunction()
{
	if (data.dataset.cols() != parameters.size())	// is numFeatures != numParameters?
		std::cout << __func__ << " Error: number of features != number of parameters" << std::endl;

	switch (modelType)
	{
	case LinearRegressionAnalytical:	// Square error cost function
	case LinearRegression:
		return 
			(0.5 / data.numExamples()) * 
			((data.dataset * parameters - data.solutions).array().pow(2)).sum();
		break;

	case LogisticRegression:			// Cost function derived from the Principle of maximum likelihood estimation (its derivative is convex).
		{
			Vector<T, Dynamic> gx = 1.0 / 
				(1.0 + Array<T, Dynamic, 1>::Constant(data.numExamples(), e).pow((-data.dataset * parameters).array()));
			
			return (1.0 / data.numExamples()) * (
				data.solutions.array() * gx.array().log() +
				(1.0 - data.solutions.array()) * (1.0 - gx.array()).log()
				).sum();
		}
		break;

	default:
		// <<< Exception
		break;
	}
};

template<typename T>
Vector<T, Dynamic> Model<T>::optimization()
{
	if (data.dataset.cols() != parameters.size())	// is numFeatures != numParameters?
		std::cout << __func__ << " Error: number of features != number of parameters" << std::endl;

	switch (modelType)
	{
	case LinearRegressionAnalytical:	// THETA = (X*T X)^-1 (X^T Y)
		{
			Matrix<T, Dynamic, Dynamic> datasetTransposed = data.dataset.transpose();
			return
				(datasetTransposed * data.dataset).completeOrthogonalDecomposition().pseudoInverse() *
				(datasetTransposed * data.solutions);
		}
		break;

	case LinearRegression:				// thetaj = thetaj - (alpha/m) sum((h(xi)-yi) xij)	(i: example) (j: feature)
		{
			return
				parameters -
				(alpha / data.numExamples()) * (
					(
						data.dataset.array().colwise() *
						(data.dataset * parameters - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
					).colwise().sum()											// Get a vector with the sum of the contents of each row
				).matrix().transpose();
		}
		break;

	case LogisticRegression:			// Like Linear Regression case, but using g(x) instead of h(x)
		{
			Vector<T, Dynamic> gx = 1.0 / (1.0 + Array<T, Dynamic, 1>::Constant(data.numExamples(), e).pow((-data.dataset * parameters).array()));

			return
				parameters -
				(alpha / data.numExamples()) * (
					(
						data.dataset.array().colwise() *
						(gx - data.solutions).array()			// Wise multiplication of a RowVector to each row of a matrix
					).colwise().sum()							// Get a vector with the sum of the contents of each row
				).matrix().transpose();
		}
		break;

	default:
		std::cout << "Non valid combination of ModelType and OptimizationType" << std::endl;
		// <<< Exception
		break;
	}
}

template<typename T>
Vector<T, Dynamic> Model<T>::pow_2(T base, Vector<T, Dynamic>& exponents)
{
	for (T& exp : exponents)
		exp = std::pow(base, exp);

	return exponents;
}

#endif