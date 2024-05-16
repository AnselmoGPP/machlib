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
	Optimization algorithms: Normal equation (lin. reg.), Batch gradient descent.
	Learning algorithm (cost function + optimization algorithm)
	Normalization: Feature scaling, Mean normalization, Learning rate, Features edition.

*/

// Constants & enums ----------

double e  = 2.71828182845904523536028747135266249;
double pi = 3.14159265358979323846264338327950288;

enum ModelType {
	LinearRegressionAnalytical,
	LinearRegression,
	LogisticRegression
};

//enum OptimizationType {
//	NormalEquation,			//!< Normal equation (anallytical solution)
//	BatchGradDescent		//!< Batch Gradient Descent
//};


// Declarations ----------

/**
	Store the dataset (examples x features) and solutions (1 x examples).
	In dataset, the first row should be all 1s (for the independent term of h)
*/
template<typename T, int Examples, int Features>
class Data
{
public:
	Data() : numExamples(Examples), numFeatures(Features) { };

	Matrix<T, Examples, Features> dataset;	//!< Matrix<T, Examples, Features>
	Vector<T, Examples> solutions;			//!< Vector<T, Examples>
	const size_t numExamples;
	const size_t numFeatures;				//!< numFeatures == numParams
	std::vector<T> range;					//!< for Feature Scaling
	std::vector<T> mean;					//!< for Mean Normalization
};


/**
	Hypothesis & Learning algorithm (cost function + optimization algorithm)
*/
template<typename T, int Examples, int Params>
class Model
{
	ModelType modelType;

	Vector<T, Examples> pow(T base, Vector<T, Examples>& exponents);

public:
	Model(ModelType modelType, double alpha)
		: modelType(modelType), alpha(alpha) { }

	Vector<T, Params> parameters;			//!< Vector<T, Params>
	double alpha;							//!< Learning rate

	T model(Vector<T, Params> features);								//!< Call h(x), where x == features you provide.
	float costFunction(Data<T, Examples, Params>& data);				//!< Compute cost function (square error cost function)
	Vector<T, Params> optimization(Data<T, Examples, Params>& data);	//!< Optimize parameters with a learning algorithm (alternatives: Conjugate gradient, BFGS, L-BFGS...)
};


template<typename T, int Examples, int Features>
class MlAlgo
{
	Data<T, Examples, Features> data;		//!< Dataset & Solutions
	Model<T, Examples, Features> h;			//!< Hypothesis & Learning algorithm

public:
	MlAlgo(float alpha, std::vector<T>& range, std::vector<T>& mean) { };
};


// Definitions ----------

template<typename T, int Examples, int Params>
T Model<T, Examples, Params>::model(Vector<T, Params> features)	// <<< reference?
{
	switch (modelType)
	{
	case LinearRegressionAnalytical:
	case LinearRegression:			// Linear function: h(x)
		return parameters.transpose() * features;
		break;

	case LogisticRegression:		// Logistic function / Sigmoid: g(x)
		return 1.0 / (1.0 + std::pow(e, -parameters.transpose() * features));
		break;
	
	default:
		// <<< Exception
		break;
	}
}

template<typename T, int Examples, int Params>
float Model<T, Examples, Params>::costFunction(Data<T, Examples, Params>& data)
{
	switch (modelType)
	{
	case LinearRegressionAnalytical:	// Square error cost function
	case LinearRegression:
		return 
			(0.5 / data.numExamples) * 
			((data.dataset * parameters - data.solutions).array().pow(2)).sum();
		break;

	case LogisticRegression:			// Cost function derived from the Principle of maximum likelihood estimation
		{
			//Vector<T, Examples> gx = 1.0 / (1.0 + pow(e, (-data.dataset * parameters)));	//<<< FIX
			//return
			//	(-data.solutions).transposed() * gx.array().log() +
			//	(- data.solutions.array() + 1.0).matrix().transposed() * (-gx.array + 1.0).log();
		}
		break;

	default:
		// <<< Exception
		break;
	}
};

template<typename T, int Examples, int Params>
Vector<T, Params> Model<T, Examples, Params>::optimization(Data<T, Examples, Params>& data)
{
	switch (modelType)
	{
	case LinearRegressionAnalytical:	// THETA = (X*T X)^-1 (X^T Y)
		{
			Matrix<T, Params, Examples> datasetTransposed = data.dataset.transpose();
			return
				(datasetTransposed * data.dataset).completeOrthogonalDecomposition().pseudoInverse() *
				(datasetTransposed * data.solutions);
		}
		break;

	case LinearRegression:				// thetaj = thetaj - (alpha/m) sum((h(xi)-yi) xij)	(i: example) (j: feature)
		return
			parameters
			- (alpha / data.numExamples) *
			(
				(
					data.dataset.array().colwise() *
					(data.dataset * parameters - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
				).colwise().sum()											// Get a vector with the sum of the contents of each row
			).matrix().transpose();
		break;

	case LogisticRegression:
		// <<<
		break;

	default:
		std::cout << "Non valid combination of ModelType and OptimizationType" << std::endl;
		// <<< Exception
		break;
	}
}

template<typename T, int Examples, int Params>
Vector<T, Examples> Model<T, Examples, Params>::pow(T base, Vector<T, Examples>& exponents)
{
	for (T& exp : exponents)
		exp = std::pow(base, exp);

	return exponents;
}

#endif