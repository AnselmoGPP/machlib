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
	Activation functions: Sigmoid (output), Hyperbolic Tangent (Tanh), Rectified Linear Unit (ReLU), Leaky ReLU, PReLU, ELU, Softmax (output).
*/

// Use binaryExpr()
// aliassing issue

// Constants & enums ----------

double e  = 2.71828182845904523536028747135266249;
double pi = 3.14159265358979323846264338327950288;

enum ModelType {
	LinearRegressionAnalytical,
	LinearRegression,			// for Regression problems (continuous values)
	LogisticRegression,			// for Classification systems (discrete values)
	None
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
	
	size_t numExamples() { return dataset.rows(); };
	size_t numParams()   { return dataset.cols(); };
	size_t numFeatures() { return dataset.cols() - 1; };

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

	float LinR_costFunction();		//!< Square error cost function for Lin. Reg.
	float LogR_costFunction();		//!< Cost function for Log. Reg. (this one can be derived using the Principle of maximum likelihood estimation.
	void LinRA_optimization();		//!< Normal equation (Ordinary least squares) for anallytic Lin. Reg.
	void LinR_optimization();		//!< Batch gradient descent for Lin. Reg.
	void LogR_optimization();		//!< Batch gradient descent for Log. Reg.

public:
	Model(ModelType modelType, double alpha, double lambda, Matrix<T, Dynamic, Dynamic>& dataset, bool normalize);

	Data<T> data;
	Vector<T, Dynamic> params;
	double alpha;								//!< Learning rate (for normalization)
	double lambda;								//!< Regularization parameter

	T model(Vector<T, Dynamic> features);		//!< Call h(x), where x == features you provide.
	float costFunction();						//!< Compute cost function (square error cost function)
	void optimize();							//!< Optimize parameters with a learning algorithm (batch gradient descent or Normal equation) (alternatives: Conjugate gradient, BFGS, L-BFGS...).

	void printParams();
};


template<typename T>
struct Layer
{
	Layer(ModelType funcType, unsigned numActUnits);

	void allocateParams(unsigned numParams);
	void saveOutput(Vector<T, Dynamic>& newOutput);	//!< Save features (not including first one)
	const Vector<T, Dynamic>& getOutput();			//!< Get features (including first one)

	ModelType funcType;					//!< Activation function type
	unsigned numActUnits;				//!< Number of activation units (each one produced by an activation function)
	Matrix<T, Dynamic, Dynamic> params;	//!< Weights (parameters)

private:
	Vector<T, Dynamic> output;			//!< Activation units + 1 (solutions)
};


template<typename T>
class DeepModel
{
public:
	DeepModel(std::vector<Layer<T>>& layersInfo, double alpha, double lambda, Matrix<T, Dynamic, Dynamic>& dataset, bool normalize);

	Data<T> data;
	std::vector<Layer<T>> layers;
	double alpha;											//!< Learning rate (for normalization)
	double lambda;											//!< Regularization parameter

	Vector<T, Dynamic> model(Vector<T, Dynamic> features);	//!< Call h(x), where x == features you provide.
	float costFunction();									//!< Compute cost function (square error cost function)
	void optimize();										//!< Optimize parameters with a learning algorithm (batch gradient descent or Normal equation) (alternatives: Conjugate gradient, BFGS, L-BFGS...).
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
	solutions = dataset.block(0, 0, dataset.rows(), 1);
	dataset.col(0).setConstant(1);			// first column is full of 1s (independent variable)
	
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

	dataset = (dataset.rowwise() - mean.transpose()).array().rowwise() / range.transpose().array();		// x = (x - mean) / range
};

template<typename T>
Model<T>::Model(ModelType modelType, double alpha, double lambda, Matrix<T, Dynamic, Dynamic>& dataset, bool normalize)
	: modelType(modelType), data(dataset, normalize), params(data.numParams()), alpha(alpha), lambda(lambda)
{
	params.setRandom();
}

template<typename T>
T Model<T>::model(Vector<T, Dynamic> features)	// < Not a reference because this allows to pass inline-created vector.
{
	// Check dimensions
	if (features.size() != data.numFeatures())
		std::cout << __func__ << "Error: wrong number of features" << std::endl;

	// Add independent variable and normalize
	Vector<T, Dynamic> normFeatures(data.numParams());
	normFeatures.row(0).setConstant(1);
	normFeatures.block(1, 0, features.rows(), 1) = features;
	normFeatures = (normFeatures - data.mean).array() / data.range.array();	// <<< aliasing issue

	// Apply model
	switch (modelType)
	{
	case LinearRegressionAnalytical:
	case LinearRegression:			// Linear function: h(x)
		return params.transpose() * normFeatures;
		break;

	case LogisticRegression:		// Logistic function / Sigmoid: g(x)
		return 1.0 / (1.0 + std::pow((T)e, -params.transpose() * normFeatures));
		break;
	
	default:
		// <<< Exception
		break;
	}
}

template<typename T>
float Model<T>::costFunction()
{
	if (data.dataset.cols() != data.numParams())
		std::cout << __func__ << " Error: wrong number of features" << std::endl;

	switch (modelType)
	{
	case LinearRegressionAnalytical:	// Square error cost function
	case LinearRegression:
		return LinR_costFunction();
		break;
	case LogisticRegression:			// Cost function derived from the Principle of maximum likelihood estimation (its derivative is convex).
		return LogR_costFunction();
		break;
	default:
		// <<< Exception
		break;
	}
};

template<typename T>
void Model<T>::optimize()
{
	if (data.dataset.cols() != data.numParams())
		std::cout << __func__ << " Error: wrong number of features" << std::endl;

	switch (modelType)
	{
	case LinearRegressionAnalytical:	// THETA = (X*T X)^-1 (X^T Y)
		LinRA_optimization();
		break;
	case LinearRegression:				// thetaj = thetaj - (alpha/m) sum((h(xi)-yi) xij)	(i: example) (j: feature)
		LinR_optimization();
		break;
	case LogisticRegression:			// Like Linear Regression case, but using g(x) instead of h(x)
		LogR_optimization();
		break;
	default:
		std::cout << "Non valid combination of ModelType and OptimizationType" << std::endl;
		// <<< Exception
		break;
	}
}

template<typename T>
void Model<T>::printParams() { std::cout << params.transpose() << std::endl; }

template<typename T>
float Model<T>::LinR_costFunction()
{
	if (lambda)
		return
			(0.5 / data.numExamples()) *
			((data.dataset * params - data.solutions).array().pow(2)).sum() +
			lambda * params.array().square().sum();
	else
		return
			(0.5 / data.numExamples()) *
			((data.dataset * params - data.solutions).array().pow(2)).sum();
}

template<typename T>
float Model<T>::LogR_costFunction()
{
	Vector<T, Dynamic> gx = 1.0 /
		(1.0 + Array<T, Dynamic, 1>::Constant(data.numExamples(), e).pow((-data.dataset * params).array()));

	if(lambda)
		return (1.0 / data.numExamples()) * (
			data.solutions.array() * gx.array().log() +
			(1.0 - data.solutions.array()) * (1.0 - gx.array()).log()
			).sum() +
			(lambda / (2 * data.numExamples())) * params.array().square().sum();
	else
		return (1.0 / data.numExamples()) * (
			data.solutions.array() * gx.array().log() +
			(1.0 - data.solutions.array()) * (1.0 - gx.array()).log()
			).sum();
}

template<typename T>
void Model<T>::LinRA_optimization()
{
	Matrix<T, Dynamic, Dynamic> datasetTransposed = data.dataset.transpose();

	if (lambda)
	{
		Matrix<T, Dynamic, Dynamic> Z = Matrix<T, Dynamic, Dynamic>::Identity(data.numParams(), data.numParams());
		Z(0, 0) = 0;
		params =
			(datasetTransposed * data.dataset + lambda * Z).inverse() *
			(datasetTransposed * data.solutions);
	}
	else
		params =
			(datasetTransposed * data.dataset).completeOrthogonalDecomposition().pseudoInverse() *
			(datasetTransposed * data.solutions);
}

template<typename T>
void Model<T>::LinR_optimization()
{
	if (lambda)
	{
		params =
			params * (1.0 - alpha * lambda / data.numExamples()) -
			(alpha / data.numExamples()) * (
				(
					data.dataset.array().colwise() *
					(data.dataset * params - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
					).colwise().sum()										// Get a vector with the sum of the contents of each row
				).matrix().transpose();
	}
	else
		params =
			params -
			(alpha / data.numExamples()) * (
				(
					data.dataset.array().colwise() *
					(data.dataset * params - data.solutions).array()	// Wise multiplication of a RowVector to each row of a matrix
				).colwise().sum()											// Get a vector with the sum of the contents of each row
			).matrix().transpose();
}

template<typename T>
void Model<T>::LogR_optimization()
{
	Vector<T, Dynamic> gx = 1.0 / (1.0 + Array<T, Dynamic, 1>::Constant(data.numExamples(), e).pow((-data.dataset * params).array()));

	if (lambda)
		params =
			params -
			(alpha / data.numExamples()) * (
				(
					data.dataset.array().colwise() *
					(gx - data.solutions).array()			// Wise multiplication of a RowVector to each row of a matrix
				).colwise().sum() +							// Get a vector with the sum of the contents of each row
				((lambda / data.numExamples()) * params).transpose().array()
			).matrix().transpose();
	else
		params =
			params -
			(alpha / data.numExamples()) * (
				(
					data.dataset.array().colwise() *
					(gx - data.solutions).array()			// Wise multiplication of a RowVector to each row of a matrix
				).colwise().sum()							// Get a vector with the sum of the contents of each row
			).matrix().transpose();
}

template<typename T>
Vector<T, Dynamic> Model<T>::pow_2(T base, Vector<T, Dynamic>& exponents)
{
	for (T& exp : exponents)
		exp = std::pow(base, exp);

	return exponents;
}

template<typename T>
Layer<T>::Layer(ModelType funcType, unsigned numActUnits)
	: funcType(funcType), numActUnits(numActUnits) 
{ 
	output.resize(numActUnits + 1);
	output[0] = 1;
}

template<typename T>
void Layer<T>::allocateParams(unsigned numParams)
{
	params.resize(numActUnits, numParams);
	params.setRandom();
}

template<typename T>
void Layer<T>::saveOutput(Vector<T, Dynamic>& newOutput)
{
	output.block(1, 0, numActUnits, 1) = newOutput;
}

template<typename T>
const Vector<T, Dynamic>& Layer<T>::getOutput() { return output; }

template<typename T>
DeepModel<T>::DeepModel(std::vector<Layer<T>>& layersInfo, double alpha, double lambda, Matrix<T, Dynamic, Dynamic>& dataset, bool normalize)
	: data(dataset, normalize), alpha(alpha), lambda(lambda)
{
	if (layers.size()) std::cerr << "Error: At least one layer is required" << std::endl;

	layers.push_back(Layer<T>(None, data.numFeatures()));

	for (unsigned i = 0; i < layersInfo.size(); i++)
	{
		layers.push_back(layersInfo[i]);
		layers[i+1].allocateParams(layers[i].numActUnits + 1);
	}
}

template<typename T>
Vector<T, Dynamic> DeepModel<T>::model(Vector<T, Dynamic> features)
{
	// Check dimensions
	if (features.size() != data.numFeatures())
		std::cout << __func__ << "Error: wrong number of features" << std::endl;

	// Add independent variable and normalize
	Vector<T, Dynamic> newFeatures = (features - data.mean.block(1, 0, data.numFeatures(), 1)).array() / data.range.block(1, 0, data.numFeatures(), 1).array();
	layers[0].saveOutput(newFeatures);

	// Apply model
	size_t i;
	for (i = 1; i < layers.size(); i++)
	{
		switch (layers[i].funcType)
		{
		case LinearRegressionAnalytical:
		case LinearRegression:			// Linear function: h(x)
			newFeatures = layers[i].params * layers[i - 1].getOutput();
			break;

		case LogisticRegression:		// Logistic function / Sigmoid: g(x)
			newFeatures = 
					1.0 / (1.0 + 
						Array<T, Dynamic, 1>::Constant(layers[i].numActUnits, e).pow(
							(- layers[i].params * layers[i - 1].getOutput()).array()
						)
					);
			break;

		default:
			// <<< Exception
			break;
		}

		layers[i].saveOutput(newFeatures);
	}

	return (layers[i - 1].getOutput()).block(1, 0, layers[i - 1].numActUnits, 1);
}

template<typename T>
float DeepModel<T>::costFunction()
{

}

template<typename T>
void DeepModel<T>::optimize()
{

}


#endif