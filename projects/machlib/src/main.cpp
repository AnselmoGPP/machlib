#include "machlib.hpp"


int main(int argc, char* argv[])
{
	Vector<int, 3> v1;
	RowVector<int, 3> v2;

	//MatrixXd a(2, 2);
	Matrix<int, Dynamic, Dynamic> m1;
	Matrix<int, 1, 3> m2{ 1, 2, 3 };
	Matrix<int, 3, 3> m3{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

	std::cout << m3 << std::endl;
}
