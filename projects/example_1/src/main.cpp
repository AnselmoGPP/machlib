#include "machlib.hpp"


int main(int argc, char* argv[])
{
	const int features = 3;		// number of features/parameters
	const int examples = 4;		// number of examples

	Data<int, features, examples> data;
	data.solutions = { 2, 4, 6, 8 };
	data.dataset << 
		1, 2, 3, 4,		// <<< should be full 1s
		3, 4, 5, 6,
		6, 7, 8, 9;

	Hypothesis<int, features, examples> hyp;
	hyp.parameters = {3, 5, 7};
	int h = hyp.compute(data.dataset.block(0, 2, 3, 1));
	int J = hyp.SquareErrorCostFunction(data);
	
	std::cout
		<< data.dataset << "\n\n"
		<< data.solutions << "\n\n"
		<< hyp.parameters << "\n\n"
		<< h << ", " << J << "\n\n";

	std::system("pause");
	return 0;
}
