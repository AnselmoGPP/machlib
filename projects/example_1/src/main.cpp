#include "machlib.hpp"


int main(int argc, char* argv[])
{
	const int features = 3;		// number of features/parameters
	const int examples = 4;		// number of examples

	Data<double, examples, features> data;
	data.solutions = { 2, 4, 6, 8 };
	data.dataset <<
		1, 3, 6,
		2, 4, 7,
		3, 5, 8,
		4, 6, 9;	// first column may be full of 1s

	Model<double, examples, features> model(LinearRegression, NormalEquation, 1.f);
	model.parameters = {3, 5, 7};
	int h = model.executeHypothesis(data.dataset.block(2, 0, 1, 3).transpose());
	int J = model.getSquareErrorCostFunction(data);
	Vector<double, features> bestParams = model.optimizeParams(data);

	std::cout << "Expected: \n  - 207 - 360 - 590.5 \n  h = 90 \n  J = 3018 \n\n";
	std::cout
		<< data.dataset << "\n\n"
		<< data.solutions << "\n\n"
		<< model.parameters << "\n\n";

	std::cout 
		<< bestParams << "\n\n"
		<< h << ", " << J << "\n\n";

	std::system("pause");
	return 0;
}
