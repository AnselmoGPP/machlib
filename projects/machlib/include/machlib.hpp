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
*/

enum MlAlgorithm {
	LinReg_NormalEc,			//!< Linear regression, Normal equation (anallytical solution)
	LinReg_BatchGradDescent,	//!< Linear regression, Batch Gradient Descent
	LogReg_BatchGradDescent		//!< Logistic regression, Batch Gradient Descent
};

enum Optimization { 
	NormalEquation,			//!< for LinearRegression models
	BatchGradientDescent	//!< for Linear & Logistic regression models
};

template<typename T, int Features>
struct MlAlgoInfo
{
	MlAlgoInfo() : numFeatures(Features), range(Features, 0), mean(Features, 0) { };

	const int numFeatures;
	size_t numExamples;
	MlAlgorithm algorithm;			//!< Hypothesis & Optimization algorithm
	std::vector<T> range;			//!< for Feature Scaling
	std::vector<T> mean;			//!< for Mean Normalization
	float alpha;					//!< Learning rate
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
};


template<typename T, int Params, int Examples>
class Hypothesis
{
public:
	Hypothesis(float alpha) : numParams(Params), alpha(alpha) { };

	RowVector<T, Params> parameters;		//!< RowVector<T, Params>
	const size_t numParams;					//!< numParams == numFeatures
	float alpha;							//!< Learning rate
	
	T executeHypothesis(Vector<T, Params> features);						//!< Call h(x), where x == features you provide.
	float SquareErrorCostFunction(Data<T, Params, Examples>& data);			//!< Compute cost function (square error cost function)
	RowVector<T, Params> optimizeParams(Data<T, Params, Examples>& data);	//!< Execute learning algorithm for optimizing parameters
};


template<typename T, int Features, int Examples>
class MLalgorithm
{
	Data<T, Features, Examples> data;				//!< Dataset & Solutions
	Hypothesis<T, Features, Examples> h;			//!< Hypothesis & Learning algorithm

public:
	MLalgorithm(float alpha) : Data(), h(alpha) { };
};


// Definitions ----------

template<typename T, int Params, int Examples>
T Hypothesis<T, Params, Examples>::executeHypothesis(Vector<T, Params> features)
{
	return parameters * features;
}

template<typename T, int Params, int Examples>
float Hypothesis<T, Params, Examples>::SquareErrorCostFunction(Data<T, Params, Examples>& data)
{
	return (0.5 / data.dataset.cols()) * ((parameters * data.dataset - data.solutions).array().pow(2)).sum();
};

template<typename T, int Params, int Examples>
RowVector<T, Params> Hypothesis<T, Params, Examples>::optimizeParams(Data<T, Params, Examples>& data)
{
	// parameter - alpha * (1/numExamples) * sum((parameters * dataset - solutions) (*) feature)
	// newVecOfParams = Vvec(params)T - alpha * (1/numExamples) * sum((params * dataset - solutions) (*) dataset)
	//                                                            |                                  |
	//                                                        sum rows						  wise mult. rows (left operand is rowVec that multiplies a matrix)

	return
		(
			parameters.transpose()
			- (alpha / data.numExamples) * 
			(
				(
					data.dataset.array().rowwise() * 
					(parameters * data.dataset - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
				).rowwise().sum()		// Get a vector with the sum of the contents of each row
			).matrix()
		).transpose();
}

#endif