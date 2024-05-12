#ifndef MACHLIB_HPP
#define MACHLIB_HPP

#include <iostream>

#include "Eigen/Dense"
/*
	#include "Eigen/Core"
	#include "Eigen/LU"
	#include "Eigen/Cholesky"
	#include "Eigen/QR"
	#include "Eigen/SVD"
	#include "Eigen/Geometry"	
	#include "Eigen/Eigenvalues"
#include "Eigen/AccelerateSupport"
#include "Eigen/CholmodSupport"
#include "Eigen/Eigen"
#include "Eigen/Householder"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/Jacobi"
#include "Eigen/KLUSupport"
#include "Eigen/MetisSupport"
#include "Eigen/OrderingMethods"
#include "Eigen/PardisoSupport"
#include "Eigen/PaStiXSupport"
#include "Eigen/QtAlignedMalloc"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"
#include "Eigen/SparseQR"
#include "Eigen/SPQRSupport"
#include "Eigen/StdDeque"
#include "Eigen/StdList"
#include "Eigen/StdVector"
#include "Eigen/SuperLUSupport"
#include "Eigen/ThreadPool"
#include "Eigen/UmfPackSupport"
*/

using Eigen::Matrix;
using Eigen::Vector;
//using Eigen::RowVector;

/*
	Conventions:
	Vectors are vertical by default.
	Dataset is examples x features.

	Terms:
	Models (h(x)): Linear regression, Logistic regression.
	Cost function: Square error.
	Optimization algorithms: Normal equation (lin. reg.), Batch gradient descent.
	Learning algorithm (cost function + optimization algorithm)
*/

// Declarations ----------

enum ModelType {
	LinearRegression,
	LogisticRegression
};

enum OptimizationType {
	NormalEquation,			//!< Normal equation (anallytical solution)
	BatchGradDescent		//!< Batch Gradient Descent
};


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
	OptimizationType optimizationType;

public:
	Model(ModelType modelType, OptimizationType optimizationType, double alpha)
		: modelType(modelType), optimizationType(optimizationType), alpha(alpha) { }

	Vector<T, Params> parameters;			//!< Vector<T, Params>
	double alpha;							//!< Learning rate

	T executeHypothesis(Vector<T, Params> features);						//!< Call h(x), where x == features you provide.
	float getSquareErrorCostFunction(Data<T, Examples, Params>& data);		//!< Compute cost function (square error cost function)
	Vector<T, Params> optimizeParams(Data<T, Examples, Params>& data);	//!< Execute learning algorithm for optimizing parameters
};


template<typename T, int Examples, int Features>
class MlAlgo
{
	Data<T, Examples, Features> data;		//!< Dataset & Solutions
	Model<T, Examples, Features> h;			//!< Hypothesis & Learning algorithm

	//MlAlgorithm algoType;			//!< Hypothesis & Optimization algorithm
	std::vector<T> range;			//!< for Feature Scaling
	std::vector<T> mean;			//!< for Mean Normalization
	float alpha;					//!< Learning rate

public:
	MlAlgo(float alpha, std::vector<T>& range, std::vector<T>& mean) { };
};


// Definitions ----------

template<typename T, int Examples, int Params>
T Model<T, Examples, Params>::executeHypothesis(Vector<T, Params> features)	// <<< reference?
{
	switch (modelType)
	{
	case LinearRegression:
		return parameters.transpose() * features;
		break;

	case LogisticRegression:
		// <<<
		break;
	
	default:
		break;
	}
}

template<typename T, int Examples, int Params>
float Model<T, Examples, Params>::getSquareErrorCostFunction(Data<T, Examples, Params>& data)
{
	switch (modelType)
	{
	case LinearRegression:
		return 
			(0.5 / data.numExamples) * 
			((data.dataset * parameters - data.solutions).array().pow(2)).sum();
		break;

	case LogisticRegression:
		// <<<
		break;

	default:
		break;
	}
};

template<typename T, int Examples, int Params>
Vector<T, Params> Model<T, Examples, Params>::optimizeParams(Data<T, Examples, Params>& data)
{
	switch (modelType)
	{
	case LinearRegression:
		{
			switch (optimizationType)
			{
			case NormalEquation:	// THETA = (X*T X)^-1 (X^T Y)
				{
					Matrix<T, Params, Examples> datasetTransposed = data.dataset.transpose();
					return
						(datasetTransposed * data.dataset).completeOrthogonalDecomposition().pseudoInverse() *
						(datasetTransposed * data.solutions);
				}
				break;

			case BatchGradDescent:	// thetaj = thetaj - (alpha/m) sum((h(xi)-yi) xij)	(i: example) (j: feature)
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

			default:
				break;
			}
		}
	case LogisticRegression:
		{
			switch (optimizationType)
			{
			case BatchGradDescent:
				break;
			}
			break;
		}

	default:
		break;
	}

	std::cout << "Non valid combination of ModelType and OptimizationType" << std::endl;
}

#endif