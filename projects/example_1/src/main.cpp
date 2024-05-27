#include "machlib.hpp"


int main1(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	// Get data
	Matrix<float, Dynamic, Dynamic> myData(5, 5);		// Columns are examples. Rows are solution + features.
	myData <<
		 7,  7, 2, 2, 3,		// This data represents: a = c^d + e | b = c + d + e
		 9,  5, 3, 2, 0,
		 6,  9, 1, 3, 5,
		 3,  4, 2, 1, 1,
		35, 14, 3, 3, 8;

	// Create model
	std::vector<Layer<float>> layers
	{
		Layer<float>(5),	// modelType, numActUnits
		Layer<float>(9),
		Layer<float>(2)
	};

	DeepModel<float> dModel(layers, LinearRegression, 0.1f, 0.f, myData, false);

	// Train model
	float J;
	for (size_t i = 0; i < 1; i++)
	{
		J = dModel.costFunction();		// get error
		//model.optimize();				// optimize parameters
		std::cout << i << ": " << J << std::endl;
	}

	// Make predictions
	Vector<float, Dynamic> input(3);
	input << 4, 3, 2;						// 4^3 + 2 = 34 | 4 + 3 + 2 = 9

	Vector<float, Dynamic> h = dModel.model(input);
	std::cout << "Prediction: \n" << h << std::endl;

	// Exit
	std::system("pause");
	return 0;
}

int main1(int argc, char* argv[])
{
	// Get data
	Matrix<float, Dynamic, Dynamic> myData(4, 3);		// Columns are examples. Rows are solution + features.
	myData <<
		4, 2, 2,	// This data represents: a = b + c
		5, 3, 2,
		4, 1, 3,
		3, 2, 1;
	
	// Create model
	Model<float> model(LinearRegression, 0.1f, 0.f, myData, false);

	// Train model
	float J;
	for (size_t i = 0; i < 1000; i++)
	{
		J = model.costFunction();		// get error
		model.optimize();				// optimize parameters
		std::cout << i << ": " << J << std::endl;
	}

	// Make predictions
	Vector<float, Dynamic> input(2);
	input << 4, 3;						// 4 + 3 = 7

	float h = model.model(input);
	std::cout << "Prediction: " << h << std::endl;
	
	// Exit
	std::system("pause");
	return 0;
}
