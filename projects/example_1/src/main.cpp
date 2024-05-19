#include "machlib.hpp"


int main(int argc, char* argv[])
{
	//const int features = 3;		// number of features/parameters
	//const int examples = 4;		// number of examples

	// Get data
	Matrix<float, Dynamic, Dynamic> myData(4, 3);		// Cols are examples. Rows are solution + features.
	myData <<
		2, 2, 4,
		2, 3, 6,
		3, 1, 3,
		1, 2, 2;
	
	Data<float> data(myData, true);

	std::cout << data.dataset << std::endl;
	std::cout << data.solutions << std::endl;
	std::cout << data.mean << std::endl;
	std::cout << data.range << std::endl;
	
	/*
	Model<double> model(LinearRegression, 1.f, data);
	model.parameters << 3, 5, 7;
	int h = model.model(data.dataset.block(2, 0, 1, 3).transpose());
	int J = model.costFunction(data);
	Vector<double, features> bestParams = model.optimization(data);

	std::cout << "Expected: \n  - 207 - 360 - 590.5 \n  h = 90 \n  J = 3018 \n\n";
	std::cout
		<< data.dataset << "\n\n"
		<< data.solutions << "\n\n"
		<< model.parameters << "\n\n"
	
		<< bestParams << "\n\n"
		<< h << ", " << J << "\n\n";
	*/
	std::system("pause");
	return 0;
}
