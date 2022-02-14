#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <cassert>

// Matrix Class
template<typename T>
class Matrix {
    // Pointer that points to the list of values
    
public:
    T *data; 
    int rows;
    int cols;
// Default Constructor
    Matrix() {
        rows = 0;
        cols = 0;
        data = nullptr;
    }
// Constructor that initializes an empty matrix if rows and columns are provided
    Matrix(int rows, int cols) : rows(rows), cols(cols){
        // std::cout << "Matrix Created" << std::endl;
    this -> data = new T[rows * cols]{0};
    }
// Constructor that fills the MAtrix with values
    Matrix(int rows, int cols, const std::initializer_list<T> &list) : Matrix(rows, cols) {
        // std::cout << "Filled Matrix Created" << std::endl;
        try {
            if (rows * cols != (int) list.size()) {
                throw "List length does not fit the matrix dimensions!";
            }
        }
        catch (const char *msg) {
            std::cout << msg << std::endl;
        }
        std::uninitialized_copy(list.begin(), list.end(), data);
    }

    // Move Constructor 
    Matrix(Matrix &&other) {
        rows = other.rows;
        cols = other.cols;

        // delete[] data;
        data = other.data;
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
        // std::cout << "Move Constructor" << std::endl;
    }

    // Copy Constructor 
    Matrix(Matrix &other) : Matrix(other.rows, other.cols) {
        //     double matrix[other.rows][other.columns] = {0};
        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                data[i * cols + j] = other.data[i * cols + j];
            }
        }
        // std::cout << "Copy Constructor" << std::endl;
    }

    // Destructor
    ~Matrix() {
        // std::cout << rows << std::endl;
        // std::cout << cols << std::endl;
        delete[] data;
        data = nullptr;
        rows = 0;
        cols = 0;
        // std::cout << "Matrix Destroyed" << std::endl;
    }

    // Copy assignment operator
    Matrix &operator=(const Matrix &other) {
        // std::cout << "Copy Assignment Operator" << std::endl;
        if (&other == this) {
            return *this;
        }
        rows = other.rows;
        cols = other.cols;

        delete[] data;

        data = new T[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            data[i] = other.data[i];
        }

        return *this;
    }

    // Move assignment operator
    Matrix &operator=(Matrix &&other) noexcept {
        // std::cout << "Move Assignment Operator" << std::endl;
        if (this != &other) {

            delete[] data;
            data = other.data;
            rows = other.rows;
            cols = other.cols;

            other.rows = 0;
            other.cols = 0;
            other.data = nullptr;

        }
        return *this;
    }

    // access operator
    T &operator[](const std::pair<int, int> &ij) {
        // std::cout<<"Access Operator"<<std::endl;

        try {
            if ((ij.first > rows) || (ij.second > cols)) {
                // std::cout<<"Rows||Cols"<<std::endl;
                // std::cout<<rows<<", "<<cols<<std::endl;
                // std::cout<<ij.first<<", "<<ij.second<<std::endl;
    
                throw "Exceeds Dimensions!";
            }
        }
        catch (const char *msg) {
            std::cout << msg << std::endl;
        }
        return data[cols * ij.first +
                    ij.second]; // throwing an exeption implies to return a T type value, so even if the pair exeeds dimension, a value will be returned.
    }


    // constant access operator
    const T &operator[](const std::pair<int, int> &ij) const {
        // std::cout << "Constant Access Operator" << std::endl;
        try {
            if ((ij.first > rows) || (ij.second > cols)) {
                throw("Const Exceeds Dimensions!");
            }
        }
        catch (const char *msg) {
            std::cout << msg << std::endl;
        }
        return data[cols * ij.first +
                    ij.second]; // throwing an exeption implies to return a T type value, so even if the pair exeeds dimension, a value will be returned.
    }


    // Scalar Multiplication Operator
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(U x) const {
        // std::cout << "Scalar Multiplication Operator" << std::endl;
        Matrix<typename std::common_type<T, U>::type> newmat(rows, cols);
        // Cycles through the values in the matrix and multiplies each by the scalar provided
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newmat[{i, j}] = data[i * cols + j] * x;
            }
        }
        return newmat;
    }

    // Matrix Multiplication Operator
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<U> &B) const {
        // std::cout << "Matrix Multiplication Operator" << std::endl;
        try{
        if (cols != B.getRows()) {
            throw "Multiplication between Matrices not Compatible!";
        }
        }
        catch (const char *msg) {
            std::cout << msg << std::endl;
        }
        Matrix<typename std::common_type<T, U>::type> newmat(rows, B.cols);

        // std::cout << newmat.getRows() << " " << newmat.getCols() << std::endl;
        // Multiplies the Matrices with the AxB * BxC = AxC
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                // T temp_sum = 0; //----------------------------------------------------------------------CHECK

                for (int k = 0; k < B.rows; k++) {

                    newmat.data[i * newmat.cols + j] += data[i * cols + k] * B.data[k * B.cols + j];
                }
                // newmat.data[i * newmat.cols + j] = temp_sum;
                // std::cout << newmat.data[i * newmat.cols + j] << std::endl;
            }
        }
        return newmat;
    }

// Addition Operator
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<U> &B) const {
        try {
            if ((((this->rows != B.rows) && ((this->rows != 1) && (B.rows != 1))) || (this->cols != B.cols))) {
                throw "Matrices not Compatible!";
            }
        }
        catch (const char *msg) {
            std::cout << msg << std::endl;
        }

        if (((rows == 1) && (B.rows != 1)) && (cols == B.cols)) {
            Matrix<typename std::common_type<T, U>::type> newmat(B.rows, B.cols);
            for (int i = 0; i < B.rows; i++) {
                for (int j = 0; j < B.cols; j++) {
                    newmat.data[i * cols + j] = data[j] + B.data[i * cols + j];
                }
            }
            return newmat;
        }

        if (((B.rows == 1) && (rows != 1)) && (cols == B.cols)) {
            Matrix<typename std::common_type<T, U>::type> newmat(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    newmat.data[i * cols + j] = data[i * cols + j] + B.data[j];
                }
            }
            return newmat;
        }

        if ((rows == B.rows) && (cols == B.cols)) {
            Matrix<typename std::common_type<T, U>::type> newmat(B.rows, B.cols);
            for (int i = 0; i < B.rows; i++) {
                for (int j = 0; j < B.cols; j++) {
                    newmat.data[i * cols + j] = data[i * cols + j] + B.data[i * cols + j];
                }
            }

            return newmat;
        } else {
            Matrix<typename std::common_type<T, U>::type> newmat;
            return newmat;
        }
    }

    // Subtraction Operator
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<U> &B) const {
        return this->template operator+(B*(-1));
    }

    Matrix transpose() const {
        // std::cout << "Transpose" << std::endl;

        Matrix newmat(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newmat.data[j * rows + i] = data[i * cols + j];
            }
        }
        return newmat;
    }

    int getRows() const { return rows; };

    int getCols() const { return cols; };

};


template<typename T>
class Layer {
public:
    // Constructor
    Layer() {}

    // Destructor
    ~Layer() {}

    // Forward Function
    virtual Matrix<T> forward(const Matrix<T> &x) = 0;

    // Backward function
    virtual Matrix<T> backward(const Matrix<T> &dy) = 0;
};

template<typename T>
class Linear : public Layer<T> {
// Private Variables
    int in_features{};
    int out_features{};
    int n_samples{};
    int seed{};
    Matrix<T> cache{};
    Matrix<T> bias{};
    Matrix<T> weights{};
    Matrix<T> bias_gradients{};
    Matrix<T> weights_gradients{};


public:

    // default constructor
    Linear() {
        in_features = out_features = n_samples = seed = 0;
    }

    // constructor
    Linear(int in_features, int out_features, int n_samples, int seed) :
            in_features(in_features), out_features(out_features), n_samples(n_samples), seed(seed) {
        // std::cout << "Linear Layer Constructed-------------------------" << std::endl;

        std::default_random_engine generator(seed);
        std::normal_distribution<T> distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);

        bias = Matrix<T>(1, out_features); //uniform
        for (int i = 0; i < out_features; i++) {
            bias[{0, i}] =  distribution_uniform(generator);
        }

        weights = Matrix<T>(in_features, out_features); //normal
        for (int i = 0; i < in_features; i++) {
            for (int j = 0; j < out_features; j++) {
                weights[{i, j}] =  distribution_normal(generator);
            }
        }

        bias_gradients = Matrix<T>(1, out_features); //zeros
        weights_gradients = Matrix<T>(in_features, out_features); //zeros
        cache = Matrix<T>(n_samples, in_features); //zeros
        // std::cout << "Weights Rows: " << weights.rows << std::endl;
    }

    // destructor
    ~Linear()= default; //watch out????

    // forward function
    virtual Matrix<T> forward(const Matrix<T> &x) override final {
        // std::cout << "Forward Linear Function-------------------------------" << std::endl;

        Matrix<T> y = x * weights + bias;
        this-> cache = x;
        // matprint(y);
        return y;
    }

    // backward function
    virtual Matrix<T> backward(const Matrix<T> &dy) override final {
        // std::cout << "Backward Linear Function----------------------------------" << std::endl;

        this->weights_gradients = cache.transpose() * dy;
        // this-> bias_gradients = dy;
        for (int i = 0; i < dy.getCols(); i++) {
            double tot = 0;
            for (int j = 0; j < dy.getRows(); j++) {
                tot += dy[{j, i}];
                this->bias_gradients[{0, i}] = tot;
            }
        }
        // matprint(bias_gradients);
        // std::cout<<dy.getRows()<<dy.getCols()<<std::endl;
        // std::cout<<weights.getRows()<<weights.getCols()<<std::endl;
        Matrix<T> result = dy * weights.transpose();
        // std::cout<<"------------------------------Weights Gradients------------------------------"<<std::endl;
        // matprint(weights_gradients);
        // std::cout<<"------------------------------Loss Gradient------------------------------"<<std::endl;
        // matprint(result);
        // std::cout<<"------------------------------Bias Gradients------------------------------"<<std::endl;
        // matprint(bias_gradients);
        return result;
    }

    // optimize function
    void optimize(T learning_rate) {
        // std::cout<<"Linear Optimizer--------------"<<std::endl;
        weights = weights - weights_gradients * learning_rate;
        // std::cout<<"------------------------------Weights Optimized------------------------------"<<std::endl;
        // matprint(weights);
        // std::cout<<"Bias Matrix"<<std::endl;
        // matprint(bias);
        // matprint(bias_gradients);
        bias = bias - bias_gradients * learning_rate;
        // std::cout<<"------------------------------Bias Optimized------------------------------"<<std::endl;
        // matprint(bias);
    }

};

template<typename T>
class ReLU : public Layer<T> {
// Private Variables
    int in_features{};
    int out_features{};
    int n_samples{};
    Matrix<T> cache{};


public:

    // default constructor
    ReLU() {
        in_features = out_features = n_samples = 0;
    }

    // constructor
    ReLU(int in_features, int out_features, int n_samples) :
            in_features(in_features), out_features(out_features),
            n_samples(n_samples) {
            this -> cache = Matrix<T>(n_samples, in_features);
            try {
            if (in_features != out_features) throw "In features and out features must be the same!";
            }
            catch (const char *msg) {
            std::cout << msg << std::endl;
        }
        // std::cout << "ReLu Layer Constructed------------------------------------" << std::endl;
    }

    // destructor
    ~ReLU() = default; //watch out????

    // forward function
    virtual Matrix<T> forward(const Matrix<T> &x) override final {
        // std::cout << "ReLu Forward-------------------------------------------------" << std::endl;
        // std::cout << x.getRows() << std::endl;
        // std::cout << x.getCols() << std::endl;
        Matrix<T> y(x.getRows(), x.getCols());
        this -> cache = x;
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.cols; j++) {
                T min_val = 0;
                y[{i,j}] = std::max(min_val, x[{i,j}]);
            }
        }
      
        return y;
    }

    // backward function
    virtual Matrix<T> backward(const Matrix<T> &dy) override final {
        // std::cout<<"ReLu Backward-----------------"<<std::endl;
        Matrix<T> dx(dy.getRows(), dy.getCols());

        for (int i = 0; i < dy.getRows(); i++) {
            for (int j = 0; j < dy.getCols(); j++) {
                T zero = 0;
                dx[{i,j}] = (dy[{i,j}] < zero ? zero : dy[{i,j}]);

                }
            }

        
        // std::cout<<"------------------------------ReLu Gradient------------------------------"<<std::endl;

        return dx;
    }

};


template<typename T>
class Net {
    // Your implementation of the Net class starts here
    // default constructor
    int in_features{};
    int hidden_dim{};
    int out_features{};
    int n_samples{};
    int seed{};
    Linear<T> linear1{};
    ReLU<T> relu{};
    Linear<T> linear2{};

public:


    Net() {
        in_features = out_features= hidden_dim = n_samples = seed= 0;
    }

    // constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed): in_features(in_features), out_features(out_features), n_samples(n_samples), seed(seed)  {
        linear1 = Linear<T>(in_features, hidden_dim, n_samples, seed);
        relu = ReLU<T>(hidden_dim, hidden_dim, n_samples);
        linear2 = Linear<T>(hidden_dim, out_features, n_samples, seed);
        // std::cout << "Net Constructed----------------" << std::endl;
    }

    // destructor
    ~Net() {} //watch out????

    // forward function
    Matrix<T> forward(const Matrix<T> &x) {
        Matrix<T> matlinear1 = linear1.forward(x);
        Matrix<T> matrelu = relu.forward(matlinear1);
        Matrix<T> matlinear2 = linear2.forward(matrelu);
        return matlinear2;
    }

    // backward function
    Matrix<T> backward(const Matrix<T> &dy) {
        Matrix<T> result_1 = linear2.backward(dy);
        Matrix<T> result_2 = relu.backward(result_1);
        Matrix<T> result_3 = linear1.backward(result_2);
        return result_3;
    }

    // optimize
    void optimize(T learning_rate) {
        // std::cout<<"Optimizing First Layer-----------"<<std::endl;
        linear1.optimize(learning_rate);
        // std::cout<<"Optimizing Second Layer------------"<<std::endl;
        linear2.optimize(learning_rate);
    }
};

//  Function to calculate the loss
template<typename T>
T MSEloss(const Matrix<T> &y_true, const Matrix<T> &y_pred) {
    // Your implementation of the MSEloss function starts here
    T sum = 0;
    int y_true_size = y_true.getRows() * y_true.getCols();
    Matrix<T> new_mat(y_true.getRows(), y_true.getCols());
    
    for (int i = 0; i < y_true.getRows(); i++) {
        for (int j = 0; j < y_true.getCols(); j++) {
            
            sum += (y_pred[{i,j}] - y_true[{i,j}]) * (y_pred[{i,j}] - y_true[{i,j}]);
        }
    }

    return sum / (y_true_size);
}


// Function to calculate the gradients of the loss
template<typename T>
Matrix<T> MSEgrad(const Matrix<T> &y_true, const Matrix<T> &y_pred) {
    // Your implementation of the MSEgrad function starts here
    Matrix<T> grad_mat(y_true.getRows(), y_true.getCols());

    for (int i = 0; i < y_pred.getRows(); i++) {
        for (int j = 0; j < y_pred.getCols(); j++) {
            T two = 2;
            grad_mat[{i,j}] = two * (y_pred[{i,j}] - y_true[{i,j}])/(y_true.getRows()*y_true.getCols());
        }
    }
    return grad_mat;
}

// Calculate the argmax
template<typename T>
Matrix<T> argmax(const Matrix<T> &y) {
    // Your implementation of the argmax function starts here
    // std::cout<<"Calculating argmax--------------"<<std::endl;
    Matrix<T> matmax(1, y.getRows());  // tRANSPOSE
    for (int i = 0; i < y.getRows(); i++) {
        int indmax = 0;
        T valmax = 0;
        for (int j = 0; j < y.cols; j++) {
            if (y[{i, j}] > valmax) { indmax = j; valmax = y[{i,j}]; }
        }
        matmax[{0, i}] = indmax;

    }
    // matprint(y);
    return matmax;
}

template<typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{ 
    Matrix<T> argmax_true = argmax(y_true);
    
    Matrix<T> argmax_pred = argmax(y_pred);
    
    T counter_true = 0;
    for (int i = 0; i < argmax_true.cols; i++) {
        if (argmax_true[{0,i}] == argmax_pred[{0,i}]) {
            T val = 1;
            counter_true += val;
        }
    }
    return counter_true / (argmax_true.cols);
}

template<typename T>
void matprint(const Matrix<T> matrix) {
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            std::cout << matrix[{i, j}] << ", ";
        }
        std::cout << std::endl;
    }

}
void argmax_test(){
Matrix<double> mat1(4,2,{3,2,3,4,4,5,9,0});
Matrix<double> mat2 = argmax(mat1);
matprint(mat2);
}
void acc_test(){
Matrix<double> mat1(2,2,{1,2,3,4});
Matrix<double> mat2(2,2,{1,2,1,0});
Matrix<double> mat3(4,2,{1,2,3,4,0,1,2,1});
Matrix<double> mat4(4,2,{1,2,1,0,1,0,0,4});
std::cout<<get_accuracy(mat3,mat4)<<std::endl;;
}
void relu_test(){
    ReLU<double> relu(3,2,3);
    Matrix<double> mat1(2,2,{1,2,4,3});
    Matrix<double> mat2(2,2,{1,2,3,4});
    Matrix<double>  mat3 = relu.forward(mat1);
    
    matprint(mat3);
}
void mat_class_test(){
    Matrix<double> C(2, 2, {1, 2, 3, 4});
    matprint(C);
    Matrix<double> A = C; // tests the copy constructor
    matprint(A);
    Matrix<double> B = std::move(C); // tests the move constructor
    matprint(B);


    Matrix<double> D, F;
    Matrix<double> E(2,2,{1,2,3,4});
    D = E; // tests the copy assign operator
    D = std::move(F); // tests the move assign operator

    matprint(D);
    // Access Operator Tests
    Matrix<double> G(2,2,{2,4,6,8});
    for (int i =0; i< G.getRows();i++){
        for(int j =0; j<G.getCols();j++)
        std::cout<<G[{i,j}];
    }


}
void mat_plus_test(){
    // Arithmetic Operator Tests
    Matrix<double> A(2,2,{2,4,6,8});
    Matrix<int> B(2,2,{2,4,6,8});
    Matrix<float> C(2,2,{2,4,6,8});

     Matrix<double> A1(1,2,{6,8});
    Matrix<int> B1(2,1,{4,6});
    Matrix<float> C1(3,1,{2,4,6});   
    auto D = A+B;
    matprint(D);
    std::cout<<typeid(D[{0,0}]).name()<<std::endl;
    auto E = A+C;
    matprint(E);
    std::cout<<typeid(E[{0,0}]).name()<<std::endl;
    auto F = B+C;
    matprint(F);
    std::cout<<typeid(F[{0,0}]).name()<<std::endl;

    auto G = A+A1;
    matprint(G);
    std::cout<<typeid(G[{0,0}]).name()<<std::endl;  
}
void mat_minus_test(){
    // Arithmetic Operator Tests
    Matrix<double> A(2,2,{2,4,6,8});
    Matrix<int> B(2,2,{2,4,6,8});
    Matrix<float> C(2,2,{2,4,6,8});
    auto D = A-B;
    matprint(D);
    std::cout<<typeid(D).name()<<std::endl;
    auto E = A-C;
    matprint(E);
    std::cout<<typeid(E).name()<<std::endl;
    auto F = B-C;
    matprint(F);
    std::cout<<typeid(F).name()<<std::endl;
}
void scalar_matmul_test(){
    int int_scalar = 2;
    double double_scalar = 2;
    float float_scalar = 2;
    Matrix<double> A(2,2,{2,4,6,8});
    Matrix<float> B(2,2,{2,4,6,8});
    Matrix<int> C(2,2,{2,4,6,8});
    std::cout<<"----------Integer Scalar Test----------"<<std::endl;
    auto int_scalar_test1  = A*int_scalar;
    
    matprint(int_scalar_test1);
    std::cout<<typeid(int_scalar_test1[{0,0}]).name()<<std::endl;
    auto int_scalar_test2  = B*int_scalar;
    matprint(int_scalar_test2);
    std::cout<<typeid(int_scalar_test2[{0,0}]).name()<<std::endl;
    auto int_scalar_test3  = C*int_scalar;
    matprint(int_scalar_test3);
    std::cout<<typeid(int_scalar_test3[{0,0}]).name()<<std::endl;

    std::cout<<"----------Double Scalar Test----------"<<std::endl;

    auto double_scalar_test1  = A*double_scalar;
    matprint(double_scalar_test1);
    std::cout<<typeid(double_scalar_test1[{0,0}]).name()<<std::endl;
    auto double_scalar_test2  = B*double_scalar;
    matprint(double_scalar_test2);
    std::cout<<typeid(double_scalar_test1[{0,0}]).name()<<std::endl;
    auto double_scalar_test3  = C*double_scalar;
    matprint(double_scalar_test3);
    std::cout<<typeid(double_scalar_test1[{0,0}]).name()<<std::endl;

    std::cout<<"----------Float Scalar Test----------"<<std::endl;

    auto float_scalar_test1  = A*float_scalar;
    matprint(float_scalar_test1);
    std::cout<<typeid(float_scalar_test1[{0,0}]).name()<<std::endl;
    auto float_scalar_test2  = B*float_scalar;
    matprint(float_scalar_test2);
    std::cout<<typeid(float_scalar_test1[{0,0}]).name()<<std::endl;
    auto float_scalar_test3  = C*float_scalar;
    matprint(float_scalar_test3);
    std::cout<<typeid(float_scalar_test1[{0,0}]).name()<<std::endl;
}
void matrix_matmul_test(){
    Matrix<double> A(2,2,{2,4,6,8});
    Matrix<float> B(2,2,{2,4,6,8});
    Matrix<int> C(2,2,{2,4,6,8});
    Matrix<double> A1(4,1,{2,4,6,8});
    Matrix<float> B1(2,1,{2,4});
    Matrix<int> C1(2,1,{4,6});
    auto double_float = A*B;
    auto float_double = B*A;
    matprint(double_float);
    std::cout<<typeid(double_float[{0,0}]).name()<<std::endl;
    matprint(float_double);
    std::cout<<typeid(float_double[{0,0}]).name()<<std::endl;

}
int main(int argc, char *argv[]) {
    // Your training and testing of the Net class starts here

    // argmax_test();
    // acc_test();
    // relu_test();
    // scalar_matmul_test();
    // matrix_matmul_test();
    mat_plus_test();
    double learning_rate = 0.0005;
    int optimizer_steps = 10000;
    int seed = 1;
    Matrix<double> xxor(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    Matrix<double> yxor(4, 2, {1, 0, 0, 1, 0, 1, 1, 0});
    int in_features = 2;
    int hidden_dim = 100;
    int out_features = 2;
    int n_samples = 8;

    Net<double> net(in_features, hidden_dim, out_features, n_samples, seed);

    for (int i = 0; i < optimizer_steps; i++) {
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Step: " << i << std::endl;
        // double acc1 = get_accuracy(yxor,xxor);
        // std::cout<<acc1<<std::endl;
        Matrix<double> fwd_step = net.forward(xxor);
        // std::cout<<"----------Forward Step----------"<<std::endl;
        // matprint(xxor);
        double loss = MSEloss(yxor, fwd_step);
        std::cout<<"LOSS"<<std::endl;
        std::cout << loss << std::endl;

        Matrix<double> gradmat = MSEgrad(yxor, fwd_step);

        Matrix<double> back_step = net.backward(gradmat);
        // std::cout<<"----------xxor----------"<<std::endl;
        // matprint(xxor);
        // std::cout<<"----------Back Step----------"<<std::endl;
        // matprint(back_step);
        // std::cout << "Optimizing--------------------------------" << std::endl;
        net.optimize(learning_rate);
        std::cout<<"Accuracy"<<std::endl;
        double acc = get_accuracy(yxor, fwd_step);
        std::cout << acc << std::endl;
    }

    return 0;

}
