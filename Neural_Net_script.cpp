#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>     

template <typename T>
class Matrix
{
    // Your implementation of the Matrix class starts here
public:
    Matrix(){
        
    }

    Matrix(int rows, int cols){
        int matrix[rows][cols] = {0};
    }
    Matrix(int rows, int cols, const std::initialize_list<T>& list){
        if (rows*cols != list.size()){// Could be an issue on this line w size
            throw "List length does not fit the matrix dimensions!";
        };
        float matrix[rows][cols];
        for (int i = 0; i< rows; i++){
            for (int j =0; j<cols; j++){
                float matrix[i][j] = list[i*cols+j];
            }
        };
        this -> rows = rows;
        this -> cols = cols;
        this -> matrix = matrix;
    }
    // Move Constructor :rows(other.rows),cols(other.cols),list(other.list)
    Matrix(Matrix&& other){
        int rows = other.rows;
        int cols = other.cols;
        // std::initialize_list<T>& list = other.list;
        float matrix = other.matrix; // fix this
        other.rows = 0;
        other.cols = 0;
        other.list = nullptr;
    }
    // Copy Constructor
    Matrix(const Matrix& other){
        float matrix[other.rows][other.columns] = {0};
        for (int i=0; i<other.rows; i++){
            for (int j=0; j< other.cols; j++){
                matrix[i][j] = other.matrix[i][j];
        }
    }
    }
    // Destructor
    ~Matrix(){

    }
    // Copy assignment operator
    Matrix operator=(const Matrix& other){
        Matrix matrix;
        matrix.rows = other.rows;
        matrix.cols = other.cols;
        matrix.list = other.list;
    }
    // Move assignment operator
    Matrix& operator=(const Matrix&& other){
        if (this !=&other){
            delete[] matrix;
            rows = 0;
            cols = 0;
        list = other.list;
        rows = other.rows;
        cols = other.cols;
        other.list = nullptr;
        other.rows = 0;
        other.cols = 0;
        };

return *this;
    }
    // access operator
    T& operator[](const std::pair<int, int>& ij) {
        if ((i >rows) || (j>cols)){
            throw "Exceeds Dimensions!";
        };
        return matrix[i][j];
    }
    // constant access operator
    const T& operator[](const std::pair<int, int>& ij) const {
        if ((i >rows) || (j>cols)){
            throw "Exceeds Dimensions!";
        };
        return matrix[i][j];
    }
    // Arithmetic Operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(U x) const {
        for (i = 0; i<row; i++){
            for (j= 0; j<cols; i++){
                matrix[i][j] = matrix[i][j]*x;
            }
        }
    }
    // arith operator 1
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(const Matrix<U>& B) const {
        if ((rows != B.rows) && (rows != B.cols)){
            throw "Matrix is not of compatible size!";
        }
            int newmat[rows][B.cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            
            for (int k = 0; k < B.rows; k++) {
                newmat[i][j] += matrix[i][k] * B.matrix[k][j];
            }

            
        }

    }
    return newmat;
    }
    // arith operator 2
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator+(const Matrix<U>& B) const {
        if (((rows = B.rows) || (B.rows = 1))&& (cols = B.cols)){
            throw "Matrices no Compatible!";
        };
        if (B.rows = 1){
        
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                matrix[i][j] += B[0][j];
            }
        }
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                matrix[i][j] += B[i][j];
            }
        }
    }
    }
    // arith operator 3
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator-(const Matrix<U>& B) const {
        if (((rows = B.rows) || (B.rows = 1))&& (cols = B.cols)){
            throw "Matrices no Compatible!";
        };
        if (B.rows = 1){
        
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                matrix[i][j] -= B[0][j];
            }
        }
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                matrix[i][j] -= B[i][j];
            }
        }
    }
    }

    // transpose
    Matrix transpose() const {...}


};

template<typename T>
class Layer
{
public:
    // Constructor 
    Layer() {}
    // Destructor
    ~Layer() {} 
    // Forward Function 
    virtual Matrix<T> forward(const Matrix<T>& x) = 0;
    // Backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) = 0;
};

template <typename T>
class Linear: public Layer<T>
{
public:
    // default constructor
    Linear() {}
    
    // constructor
    Linear(int in_features, int out_features, int n_samples, int seed) {
        Matrix<int> bias(1,out_features); //uniform
        Matrix<int> weights(in_features,out_features); //normal
        Matrix<int> bias_gradients(1,out_features); //zeros
        Matrix<int> weights_gradients(in_features,out_features); //zeros
        Matrix<int> cache(n_samples,in_features); //zeros
    }

    // destructor
    ~Linear(){}

    // forward function
    virtual Matrix<T> forward(const Matrix<T>& x) override final {

    }

    // backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {

    }

    // transpose function


    // optimize function
    void optimize(T learning_rate) {
        
    }

};
template <typename T>
class ReLU: public Layer<T>
{
    // Your implementation of the ReLU class starts here
    // constructor
    // destructor
    // forward function
    // backward function
};

template <typename T>
class Net 
{
    // Your implementation of the Net class starts here
    // constructor
    // destructor
    // forward function
    // backward function
    // optimize
};
// fff
// Function to calculate the loss
template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEloss function starts here
    float sum = 0;
    double new_mat[y_true.size()] = {};
    for (int i = 0; i<y_true.size(); i++){
        new_mat[i] = (y_pred[i]-y_true[i])*(y_pred[i]-y_true[i]);
    }
    for (int i = 0; i<y_true.size(); i++){
        sum+=new_mat[i];
    }
    return sum/(y_true.size());
}

// Function to calculate the gradients of the loss
template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEgrad function starts here
}

// Calculate the argmax 
template <typename T>
Matrix<T> argmax(const Matrix<T>& y) 
{
    // Your implementation of the argmax function starts here
}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    // Your implementation of the get_accuracy starts here
}

int main(int argc, char* argv[])
{
    // Your training and testing of the Net class starts here


    return 0;
}
