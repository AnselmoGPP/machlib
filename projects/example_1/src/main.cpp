#include "machlib.hpp"


int main(int argc, char* argv[])
{
	//const int features = 3;		// number of features/parameters
	//const int examples = 4;		// number of examples

	// Get data
	Matrix<float, Dynamic, Dynamic> myData(4, 3);		// Cols are examples. Rows are solution + features.
	myData <<
		4, 2, 2,
		6, 3, 2,
		3, 1, 3,
		2, 2, 1;

	Vector<float, Dynamic> input(2);
	input << 4, 3;
	std::cout << "-------" << std::endl;
	Model<float> model(LinearRegression, 1.f, myData, true);
	model.parameters.setRandom();
	std::cout << "-------" << std::endl;
	int h = model.model(myData.block(2, 1, 1, 2).transpose());	std::cout << "-------" << std::endl;
	int J = model.costFunction();	std::cout << "-------" << std::endl;
	Vector<float, Dynamic> bestParams = model.optimization();	std::cout << "-------" << std::endl;

	std::cout << "Dataset: \n" << model.data.dataset << std::endl;
	std::cout << "Solutions: \n" << model.data.solutions << std::endl;
	std::cout << "Mean: \n" << model.data.mean << std::endl;
	std::cout << "Range: \n" << model.data.range << std::endl;

	std::cout << "Expected: \n  - 207 - 360 - 590.5 \n  h = 90 \n  J = 3018 \n\n";
	std::cout
		<< model.data.dataset << "\n\n"
		<< model.data.solutions << "\n\n"
		<< model.parameters << "\n\n"
	
		<< bestParams << "\n\n"
		<< h << ", " << J << "\n\n";
	
	std::system("pause");
	return 0;
}
