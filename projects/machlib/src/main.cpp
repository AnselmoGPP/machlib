#include "machlib.hpp"

int main(int argc, char* argv[])
{
	// ----- Eigen quick guide -----
	
	// The following examples are based on: https://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html
	// Quick reference guide: https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

	// Getting started
	{
		MatrixXd m = MatrixXd::Random(3, 3);
		m = (m + MatrixXd::Constant(3, 3, 1.5)) * 10;
		std::cout << m << std::endl;

		VectorXd v(3);
		v << 1, 2, 3;
		std::cout << m * v << std::endl;
	}

	// Matrix class
	{
		// Creation
		Matrix<float, 4, 4> m4f;
		Matrix<float, 3, 1> v3f;
		Matrix<float, 1, 2> rowV2f;						// Vectors are matrices under the hood
		Matrix<double, Dynamic, Dynamic> mXd;			// Dynamic size: size unknown at compile time (Fixed size: size known at compile time).
		Matrix<float, 3, 6, RowMajor, 3, 6> m;		// Matrix<type, rowsAtCompileTime, colsAtCompileTime, OptionsBitField (storage order...), maxRowsAtCompileTime, maxColsAtCompileTime>.
		Matrix<float, Dynamic, Dynamic, 0, 3, 4> mm;	// Set max maxColsAtCompileTime and maxRowsAtCompileTime to prevent dynamic memory allocation. This example uses a plain array of 12 floats without dynamic memory allocation.

		// typedefs:
		MatrixXi t1;	// MatrixNt  == Matrix<type, N, N>
		MatrixX3i t2;	// MatrixXNt == Matrix<types, Dynamic, N>
		Matrix4Xd t3;	// MatrixNXt == Matrix<type, N, Dynamic>
		Vector2f t4;	// VectorNt == Matrix<type, N, 1>
		RowVector3d t5;	// RowVectorNt == Matrix<type, 1, N>
		//     where N = 2, 3, 4, or X (Dynamic).
		//     where t = i (int), f (float), d (double), cf (complex<float>), cd (complex<double>).
	
		Matrix3f m1;			// default constructor doesn't dynamically allocate memory nor initializes coefficients.
		MatrixXf m2(10, 15);	// Array allocation (10x15) but no initialization of coefficients
		VectorXf v1(20);		// Array allocation (20) but no initialization of coefficients

		// Initialization
		Vector2d v3(1.0, 2.0);
		Matrix<int, 5, 1> m3{ 1, 2, 3, 4, 5 };
		Matrix<int, 1, 5> m4 = { 1, 2, 3, 4, 5 };
		MatrixXi m5{ {1, 2}, {3, 4} };					// Group rows together
		Matrix<double, 2, 3> b{ {2, 3, 4}, {5, 6, 7} };
		VectorXd v4{ {1.0, 2.0, 3.0} };					// Column vectors can be initialized from a single row
		RowVectorXd v5{ {1.0, 2.0, 3.0} };
		m5 << 1, 2, 3, 4;								// Comma-initialization

		// Coefficient accessors
		m5(0, 1) = m5(1, 1);
		v5(1) = v5(2);
		v5[1] = v5[2];		// Operator [] overloaded only for vectors

		// Resizing
		MatrixXd m6(2, 5);
		m6.resize(4, 3);			// Resize a dynamic-size matrix/vector. Coefficient's values may change if the actual matrix size don't.
		m6.conservativeResize(3, 3);// Like resize(), but coefficient's values don't change
		m6.size();
		m6.rows();
		m6.cols();

		MatrixXf m7(2, 2);
		MatrixXf m8(3, 3);
		m7 = m8;		// m7 is automatically resized to match m8 size (m7 must have dynamic size).

		// Fixed vs. Dynamic sizes
		/*
		Fixed sizes:
			Use it for very small sizes (~16<).
			Avoids dynamic memory allocation and loops unroll.
			Internally, a fixed size matrix is just a plain array, which has zero runtime cost (Matrix4f m == float m[16]).
			Requires knowing sizes at compile time.
		Dynamic sizes:
			Use it for larger sizes (~32>).
			For very large matrices, creating a fixed-size one in a function may cause a stack overflow since Eigen will try to allocate the array as a local variable in the stack (smaller than heap).
			It's allocated on the heap (MatrixXf m(2, 3) == float *m = new float[6]).
			It stores the number of rows and columns as member variables
			Eigen will try harder to vectorize (use SIMD instructions).
		*/
	}

	// Matrix & vector arithmetic
	{
		// Addition & Subtraction
		Matrix2d m1, m2;
		m1 + m2;
		m1 - m2;
		m1 += m2;
		Vector3d v1, v2;
		-v1 + v2 - v1;

		// Scalar multiplication & division
		double c = 2.5;
		m1* c;
		c* v1;
		v1 *= 2;

		// Optimized expression templates
		m1 = 3 * m1 + 4 * m2 + 5 * m2;	// Optimization: The actual computation happens when the whole expression is evaluated (typically in operator =).

		// Transposition & Conjugation
		m1.transpose();			// returns the transpose
		m1.conjugate();			// returns the conjugate
		m1.adjoint();			// returns the adjoint
		m1 = m2.transpose();	// Correct
		m1 = m1.transpose();	// Error: Aliasing issue (Unexpected result. The result is writen into m1 before finishing the transpose).
		m1.transposeInPlace();	// Use this instead.
		m1.adjointInPlace();

		// Matrix*Matrix & Matrix*Vector
		Vector2d v3;
		m1 * m2;
		m1 * v3;
		m1 = m1 * m1;			// No error here (a temporary is introduced internally).
		v3.noalias() += m1 * v3;// Avoid the temporary if you know the matrix product can be safely evaluated into the destination matrix without aliasing issue.
	
		// Dot & Cross product
		v1.cross(v2);			// Cross product is only for vectors of size 3.
		v1.dot(v2);
		v1.adjoint() * v2;		// = dot product (as 1x1 matrix).

		// Reduction operations
		m1.sum();				// Sum all the elements
		m1.prod();				// Multiply all elements
		m1.trace();				// Sum of diagonal coefficients
		m1.diagonal().sum();	// = m1.trace()
		m1.mean();

		std::ptrdiff_t i, j;
		m1.minCoeff();			// get min/max coefficient
		m1.maxCoeff();
		m1.minCoeff(&i, &j);	// and get coordinates
		v1.maxCoeff(&i);
		
		// Operations validity
		/*
		Eigen checks the validity of the operations that you perform. 
		When possible, it checks them at compile time, producing compilation errors. 
		If check can't be performed at compile time (example: when checking dynamic sizes) it uses runtime assertions 
		(i.e. in "debug mode" the program will abort with error message when executing an illegal operation, 
		but will probably crash if assertions are turned off).
		*/
	}

	// Array class and coefficient-wise operations
	{
		// Array class: Provides general-purpose arrays and an easy way to perform coefficient-wise operations. Requires the same parameters as the Matrix class.
		// Matrix class: Intended for linear algebra operations.

		// Array typedefs
		Array<float, Dynamic, 1> t1;		// ArrayXf
		Array<float, 3, 1> t2;				// Array3f
		Array<double, Dynamic, Dynamic> t3;	// ArrayXXd
		Array<double, 3, 3> t4;				// Array33d
		
		// Creation
		ArrayXf a1 = ArrayXf::Random(4);
		ArrayXXf a2(2, 2);
		a2(1, 0) = 3.0;
		a2 << 1, 2, 3, 4;

		// Coefficient-wise operations
		ArrayXXf a3(2, 2);
		a2 + a3;
		a2 - 5;
		a2 * a3;
		a2.abs();
		a2.sqrt();
		a2.min(a3);
		a2.min(a2.abs().sqrt());

		// Conversions array-matrix
		// You cannot apply Matrix operations on arrays, or Array operations of matrices.
		// Use matrices for linear algebraic operations. Use arrays for coefficient-wise operations.
		// Convert to matrix or array with matrix() or array() (both can be used as rvalue and lvalue).
		// Mixing matrices and arrays in an expression is forbidden (example: to use + operator both have to be matrices or arrays).
		// Exception: the assignment operator can assign a matrix to an array, or viceversa.
		Matrix2f m1, m2, m3;
		m1 = m2 * m3;					// Matrix product
		m1 = m2.array() * m3.array();	// coefficient-wise multiplication
		m1 = m2.cwiseProduct(m3);		// equivalent
		m1 = m2.array() + 5;			// coefficient-wise sum (matrix + scalar)
		m1 = (m2.array() + 5).matrix() * m3;			// Coefficient-wise addition > Matrix product.
		m1 = (m2.array() * m3.array()).matrix() * m2;	// Coefficient-wise multiplication > Matrix product.
	}
}
