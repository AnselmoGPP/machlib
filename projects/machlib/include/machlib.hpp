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
using Eigen::Dynamic;

/*
template<typename T, int Rows, int Cols>
class MlMatrix
{
public:
	MlMatrix() { };

	Matrix<T, Rows, Cols> matrix;
};
*/

/**
	Store the dataset (features x examples) and solutions (1 x examples).
	In dataset, the first row should be all 1s (for the independent term of h)
*/
template<typename T, int Features, int Examples>
class Data
{
public:
	Data() : features(Features), examples(Examples) { };

	Matrix<T, Features, Examples> dataset;	//!< Matrix<T, Features, Examples>
	RowVector<T, Examples> solutions;		//!< RowVector<T, Examples>
	const int features;
	const int examples;
};


template<typename T, int Params, int Examples>
class Hypothesis
{
public:
	Hypothesis() { };

	RowVector<T, Params> parameters;		//!< RowVector<T, Params>
	
	T compute(Vector<T, Params> features) { return parameters * features; }

	T SquareErrorCostFunction(Data<T, Params, Examples>& data)
	{ return (0.5 / data.dataset.cols()) * ((parameters * data.dataset - data.solutions).array().pow(2)).sum(); };
	//{ return (0.5 / data.dataset.cols()) * (parameters * data.dataset - data.solutions).array().pow(2); };
};


template<typename T, int Params>
class SquareErrorCostFunction
{
public:
	SquareErrorCostFunction() { };

	//double compute(
	//	RowVector<T, Params>& parameters, 
	//	Matrix<T, Features, Examples>& dataset,
	//	) 
	//{ return (0.5 / dataset.cols()) * (parameters * dataset - solutions)^2; };
};


class LearnAlgorithm
{

};


#endif