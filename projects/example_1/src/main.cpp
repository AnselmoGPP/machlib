#include "machlib.hpp"


int main(int argc, char* argv[])
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
