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

//using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Vector;
using Eigen::RowVector;
//using Eigen::Dynamic;

// Declarations ----------

/*
	Linear regression (GD, Normal equation)
	Logistic regression (GD)
	Model h(x)
	Cost function
	Optimization algorithm (Normal ec., Batch Grad. Descent)
	Learning algorithm (cost function + optimization algorithm)
*/

//enum MlAlgorithm {
//	LinReg_NormalEc,			//!< Linear regression, Normal equation (anallytical solution)
//	LinReg_BatchGradDescent,	//!< Linear regression, Batch Gradient Descent
//	LogReg_BatchGradDescent		//!< Logistic regression, Batch Gradient Descent
//};

enum ModelType {
	LinearRegression,
	LogisticRegression
};

enum OptimizationType {
	NormalEquation,			//!< Normal equation (anallytical solution)
	BatchGradDescent		//!< Batch Gradient Descent
};


/**
	Store the dataset (features x examples) and solutions (1 x examples).
	In dataset, the first row should be all 1s (for the independent term of h)
*/
template<typename T, int Features, int Examples>
class Data
{
public:
	Data() : numFeatures(Features), numExamples(Examples) { };

	Matrix<T, Features, Examples> dataset;	//!< Matrix<T, Features, Examples>
	RowVector<T, Examples> solutions;		//!< RowVector<T, Examples>
	const size_t numFeatures;				//!< numFeatures == numParams
	const size_t numExamples;
	std::vector<T> range;					//!< for Feature Scaling
	std::vector<T> mean;					//!< for Mean Normalization
};


/**
	Hypothesis & Learning algorithm (cost function + optimization algorithm)
*/
template<typename T, int Params, int Examples>
class Model
{
	ModelType modelType;
	OptimizationType optimizationType;
	/*
	// Hypothesis/Model
	T executeLinReg();
	T executeLogReg();

	// Cost function
	double squareErrorLinReg();
	double squareErrorLogReg();

	// Optimization
	RowVector<T, Params> optimizeLinReg();
	RowVector<T, Params> optimizeLogReg();
	RowVector<T, Params> normalEquation();
	*/
public:
	Model(ModelType modelType, OptimizationType optimizationType, double alpha)
		: modelType(modelType), optimizationType(optimizationType), alpha(alpha) { }

	RowVector<T, Params> parameters;		//!< RowVector<T, Params>
	double alpha;							//!< Learning rate

	T executeHypothesis(Vector<T, Params> features);						//!< Call h(x), where x == features you provide.
	float getSquareErrorCostFunction(Data<T, Params, Examples>& data);		//!< Compute cost function (square error cost function)
	RowVector<T, Params> optimizeParams(Data<T, Params, Examples>& data);	//!< Execute learning algorithm for optimizing parameters
};


template<typename T, int Features, int Examples>
class MlAlgo
{
	Data<T, Features, Examples> data;				//!< Dataset & Solutions
	Model<T, Features, Examples> h;			//!< Hypothesis & Learning algorithm

	//MlAlgorithm algoType;			//!< Hypothesis & Optimization algorithm
	std::vector<T> range;			//!< for Feature Scaling
	std::vector<T> mean;			//!< for Mean Normalization
	float alpha;					//!< Learning rate

public:
	MlAlgo(float alpha, std::vector<T>& range, std::vector<T>& mean) { };
};


// Definitions ----------

template<typename T, int Params, int Examples>
T Model<T, Params, Examples>::executeHypothesis(Vector<T, Params> features)	// <<< reference?
{
	switch (optimizationType)
	{
	case NormalEquation:	// THETA = (X*T X)^-1 (X^T Y)
		//return 
		//	(features.dataset.transpose() * features.dataset).inverse() *	
		//	features.dataset.transpose() * 
		//	features.solutions;

		break;

	case BatchGradDescent:
		return parameters * features;
		break;
	
	default:
		break;
	}
}

template<typename T, int Params, int Examples>
float Model<T, Params, Examples>::getSquareErrorCostFunction(Data<T, Params, Examples>& data)
{
	switch (modelType)
	{
	case LinearRegression:
		return 
			(0.5 / data.dataset.cols()) * 
			((parameters * data.dataset - data.solutions).array().pow(2)).sum();
		break;

	case LogisticRegression:
		break;

	default:
		break;
	}
};

template<typename T, int Params, int Examples>
RowVector<T, Params> Model<T, Params, Examples>::optimizeParams(Data<T, Params, Examples>& data)
{
	switch (modelType)
	{
	case LinearRegression:
		return
			(
				parameters.transpose()
				- (alpha / data.numExamples) * 
				(
					(
						data.dataset.array().rowwise() * 
						(parameters * data.dataset - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
					).rowwise().sum()											// Get a vector with the sum of the contents of each row
				).matrix()
			).transpose();
		break;

	case LogisticRegression:
		break;

	default:
		break;
	}
}

/*
template<typename T, int Params, int Examples>
T Model<T, Params, Examples>::executeLinReg()
{

}

template<typename T, int Params, int Examples>
T Model<T, Params, Examples>::executeLogReg()
{

}

template<typename T, int Params, int Examples>
double Model<T, Params, Examples>::squareErrorLinReg()
{

}

template<typename T, int Params, int Examples>
double Model<T, Params, Examples>::squareErrorLogReg()
{

}

template<typename T, int Params, int Examples>
RowVector<T, Params> Model<T, Params, Examples>::optimizeLinReg()
{

}

template<typename T, int Params, int Examples>
RowVector<T, Params> Model<T, Params, Examples>::optimizeLogReg()
{

}

template<typename T, int Params, int Examples>
RowVector<T, Params> Model<T, Params, Examples>::normalEquation()
{

}
*/
#endif