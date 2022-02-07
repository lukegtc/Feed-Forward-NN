#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>     
 
template <typename T>
class Matrix {
    // Your implementation of the Matrix class starts here

public:
    int rows;
    int cols;
    // std::initializer_list<T>& list;
    double* data;
    // std::vector<T>& matrix;
    Matrix(){
        rows = 0;
        cols = 0;
        data = nullptr;
    }

    Matrix(int rows, int cols):data(new double[rows*cols]{0}){
        this -> rows = rows;
        this -> cols = cols;    
    }

    Matrix(int rows, int cols, const std::initializer_list<T>& list):Matrix(rows,cols){
        std::cout<<"Matrix Created"<<std::endl;
        try {
            if (rows*cols != (int)list.size()){
                throw "List length does not fit the matrix dimensions!";}
        }
        catch (const char* msg) {
            std::cout << msg << std::endl;
        }
        std::uninitialized_copy(list.begin(),list.end(),data);
    }

    // Move Constructor :rows(other.rows),cols(other.cols),list(other.list)
    Matrix(Matrix&& other):rows(other.rows),cols(other.cols),data(other.data){
        // int rows = other.rows;
        // int cols = other.cols;
        // std::initialize_list<T>& list = other.list;
        // double matrix = other.matrix; // fix this
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
        std::cout<<"Move Constructor"<<std::endl;
    }

    // Copy Constructor
    Matrix(const Matrix& other):Matrix(other.rows,other.cols){
    //     double matrix[other.rows][other.columns] = {0};
        for (int i=0; i<other.rows; i++){
            for (int j=0; j< other.cols; j++){
                data[i*cols + j] = other.data[i*cols+j];
                }
            }
    }

    // Destructor
    ~Matrix(){
    delete[] data;
    data = nullptr;
    rows = 0;
    cols = 0;
    }

    // Copy assignment operator
    Matrix operator=(const Matrix& other){
        std::cout<<"Copy Assignment Operator"<<std::endl;
        Matrix matrix;
        matrix.rows = other.rows;
        matrix.cols = other.cols;
        matrix.data = other.data;
        return *this;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other){
        if (this !=&other){
        std::cout<<"Move Assignment Operator"<<std::endl;
        delete[] data;
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        }
        return *this;
    }

    // access operator
    T& operator[](const std::pair<int, int>& ij){
        // std::cout<<"Access Operator"<<std::endl;
        try {
        if ((ij.first >rows) || (ij.second>cols)){
            throw "Exceeds Dimensions!";}
        }
        catch (const char* msg) {
            std::cout << msg << std::endl;
        } 
        return  data[cols*ij.first+ij.second]; // throwing an exeption implies to return a T type value, so even if the pair exeeds dimension, a value will be returned.
    }
   

    // constant access operator
    const T& operator[](const std::pair<int, int>& ij) const {
        std::cout<<"Constant Access Operator"<<std::endl;
       try {
            if ((ij.first >rows) || (ij.second>cols)){
                throw "Exceeds Dimensions!";}
        }
        catch (const char* msg) {
            std::cout << msg << std::endl;
        } 
        return  data[cols*ij.first+ij.second]; // throwing an exeption implies to return a T type value, so even if the pair exeeds dimension, a value will be returned.
    }

    // Scalar Multiplication Operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(U x) const {
        std::cout<<"Scalar Multiplication Operator"<<std::endl;
        Matrix<T> newmat(rows,cols);
        for (int i = 0; i<rows; i++){
            for (int j = 0; j<cols; j++){
                newmat[{i,j}] = data[i*cols+j] * x; } }
        return newmat;
    }

    // Matrix Multiplication Operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(const Matrix<U>& B) const {
        std::cout<<"Matrix Multiplication Operator"<<std::endl;
        std::cout<<cols<<std::endl;
        std::cout<<B.rows<<std::endl;
        try {
            if (cols != B.rows){
                throw "Matrices not Compatible!";}
        }
        catch (const char* msg) {
            std::cout << msg << std::endl;
        } 
        Matrix newmat(rows,B.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < B.cols; j++) {
            
                for (int k = 0; k < B.rows; k++) {
                    newmat.data[i*newmat.cols+j] += data[i*cols+k] * B.data[k*B.cols+j];
                }
            }
        }
    return newmat;
    }

    // Matrix Addition Operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator+(const Matrix<U>& B) const {
        std::cout<<"Matrix Addition Operator"<<std::endl;

        try {
            if (((rows != B.rows) && (rows != 1))|| (cols != B.cols)){
                throw "Matrices not Compatible!";}
        }
        catch (const char* msg) {
            std::cout << msg <<std::endl;
    
        }
        Matrix newmat(B.rows,B.cols);
        if (rows == 1){
        
            for (int i=0; i<B.rows; i++){
                for (int j=0; j< B.cols; j++){
                    newmat.data[i*cols + j] = data[j] + B.data[i*cols+j];
                    }
                }
            };
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                newmat.data[i*cols + j] = data[i*cols + j] + B.data[i*cols + j];
                }
            };
    return newmat;
    }

    // Matrix Subtraction Operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator-(const Matrix<U>& B) const {
            std::cout<<"Matrix Subtraction Operator"<<std::endl;

        try {
            if (((rows != B.rows) && (rows != 1))|| (cols != B.cols)){
                throw "Matrices not Compatible!";}
        }
        catch (const char* msg) {
            std::cout << msg << std::endl;
        }
        Matrix newmat(B.rows,B.cols);
        if (rows == 1){
        
            for (int i=0; i<B.rows; i++){
                for (int j=0; j< B.cols; j++){
                    newmat.data[i*cols + j] = data[j] - B.data[i*cols+j];
                    }
                }
            };
        for (int i=0; i<B.rows; i++){
            for (int j=0; j< B.cols; j++){
                newmat.data[i*cols + j] = data[i*cols + j] - B.data[i*cols + j];
                }
            };
        return newmat;
    }

    // transpose
    Matrix transpose() const {
                std::cout<<"Transpose"<<std::endl;

        Matrix newmat(cols,rows);
        for (int i=0; i<rows; i++){
            for (int j=0; j< cols; j++){
                newmat.data[j*rows + i] = data[i*cols + j];
                }
            } 
    return newmat;  
    }

    int getRows() const {return rows;};
    int getCols() const {return cols;};
    
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

int in_features;
int out_features;
int n_samples;
public:

    Matrix<T> cache;
    Matrix<T> bias;
    Matrix<T> weights;
    Matrix<T> bias_gradients;
    Matrix<T> weights_gradient;


    // default constructor
    Linear() {}
    
    // constructor
    Linear(int in_features, int out_features, int n_samples, int seed):
    in_features(in_features), out_features(out_features), n_samples(n_samples) {
        std::cout<<"Linear Layer Constructed"<<std::endl;

        std::default_random_engine        generator(seed);
        std::normal_distribution<T>       distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);

        Matrix<T> bias(1,out_features); //uniform
        for (int i=0; i<out_features; ++i) {
            
            bias[{0,i}] = distribution_uniform(generator);
            
        }

        Matrix<T> weights(in_features,out_features); //normal
        for (int i=0; i<in_features; i++) {
            for (int j=0; j<out_features; j++) {
                weights[{i,j}] = distribution_normal(generator);
            }
        }

        Matrix<T> bias_gradients(1,out_features); //zeros
        Matrix<T> weights_gradient(in_features,out_features); //zeros
        Matrix<T> cache(n_samples,in_features); //zeros
    }

    // destructor
    ~Linear(){} //watch out????

    // forward function
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        Matrix<T> y=x*weights+bias;
        cache = x;
        return y;
    }

    // backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        Matrix<T> dx = dy*weights.transpose();
        return dx;
    }

    // optimize function
    void optimize(T learning_rate) {
        weights = weights - weights_gradient*learning_rate ;
        bias = bias - bias_gradients*learning_rate ;
    }

};

template <typename T>
class ReLu: public Layer<T>
{
int in_features;
int out_features;
int n_samples;

public:

    Matrix<T> cache;
    Matrix<T> bias;
    Matrix<T> weights;
    Matrix<T> bias_gradients;
    Matrix<T> weights_gradient;


    // default constructor
    ReLu() {}
    
    // constructor
    ReLu(int in_features, int out_features, int n_samples) {
    std::cout<<"ReLu Layer Constructed"<<std::endl;
        Matrix<T> cache(n_samples,in_features); //zeros
        this -> in_features=in_features;
        this -> out_features=out_features;
        this -> n_samples=n_samples;
    }

    // destructor
    ~ReLu(){} //watch out????

    // forward function
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        Matrix<T> y;
        for (int i=0; i<x.rows; i++) {
            for(int j=0; j<x.cols; j++){
                
            y[{i,j}] = std::max(0.0,x[{i,j}]);}}
        return y;
    }

    // backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        Matrix<T> dx;
        for (int i=0; i<dy.rows; i++) {
            for (int j=0; j<dy.cols; j++)
            if (
                dy[{i,j}]<0){dx[{i,j}] = 0;}
            else {dx[{i,j}] = 1;}}
        return dx;
    }
    
};


 
template <typename T>
class Net 
{
    // Your implementation of the Net class starts here
    // default constructor
    public:
        Linear<T> linear1;
        ReLu<T> relu;
        Linear<T> linear2;

    Net() {}
    // constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed) {
        Linear<T> linear1(in_features, out_features, n_samples, seed);
        ReLu<T> relu(in_features, out_features, n_samples);
        Linear<T> linear2(in_features, out_features, n_samples, seed);

    }
    // destructor
    ~Net(){} //watch out????

    // forward function
    Matrix<T> forward(const Matrix<T>& x) {
        Matrix<T> matlinear1 = linear1.forward(x);
        Matrix<T> matrelu = relu.forward(matlinear1);
        Matrix<T> matlinear2 = linear2.forward(matrelu);
        return matlinear2;

    }

    // backward function
    Matrix<T> backward(const Matrix<T>& dy) {       
        Matrix<T> matlinear2 = linear2.backward(dy);
        Matrix<T> matrelu = relu.backward(matlinear2);
        Matrix<T> matlinear1 = linear1.backward(matrelu);
        return matlinear1;
    }

    // optimize
    void optimize(T learning_rate) {
        linear1.optimize(learning_rate);
        linear2.optimize(learning_rate);
    }
};

//  Function to calculate the loss
template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
     // Your implementation of the MSEloss function starts here
    float sum = 0;
    int y_true_size = y_true.getRows() * y_true.getCols();
    Matrix<T> new_mat(y_true.getRows(),y_true.getCols());
    for (int i = 0; i< y_true.getRows(); i++){
        for (int j = 0; j<  y_true.getCols(); j++){
        new_mat[{i,j}] = (y_pred[{i,j}]-y_true[{i,j}])*(y_pred[{i,j}]-y_true[{i,j}]);
    }}
    for (int i = 0; i< y_true.getCols() ; i++){
        for (int j = 0; j< y_true.getCols() ; j++)
        sum+=new_mat[{i,j}];
    }
    return sum/(y_true_size);
}


// Function to calculate the gradients of the loss
template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEgrad function starts here
    Matrix<T> grad_mat;
    for (int i = 0; i<y_pred.getRows(); i++){
        for (int j = 0; j<y_pred.getCols(); j++){
        grad_mat[{i,j}] = 2*(y_pred[{i,j}]-y_true[{i,j}]);
    }}
    return grad_mat;
}

 // Calculate the argmax 
template <typename T>
Matrix<T> argmax(const Matrix<T>& y) 
{
    // Your implementation of the argmax function starts here
    Matrix<T> matmax(1,y.getCols());
    for (int i=0; i<y.rows; i++){
            T indmax;
            T valmax = 0;
            for (int j=0; j< y.cols; j++){
                if (y[{i,j}]> valmax) {indmax = j;}
                }
            matmax[{0,i}] = indmax;

}
return matmax;}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    // Your implementation of the get_accuracy starts here
    Matrix<T> matmax_true = argmax(y_true);
    Matrix<T> matmax_pred = argmax(y_pred);
    double tot;
    for (int i=0; i<y_true.getRows(); i++){
        for (int j = 0; j<y_true.getCols(); j++){
            tot +=(matmax_true[{i,j}]-matmax_pred[{i,j}])/matmax_true[{i,j}];

        }
    }
    return tot/(y_true.getRows()*y_true.getCols());
}



int main(int argc, char* argv[])
{
    // Your training and testing of the Net class starts here

    // Matrix<double> mat1(2,2,{1,2,3,4});
    // std::pair<int, int> pair1(0,1);
    // int x = 3;
    // Matrix<double> mat2 = mat1*x;
    // Matrix<double> mat3 = mat1*mat1;
    // Matrix<double> mat4 = mat3-mat1;
    // Matrix<double> mat5 = mat1.transpose();
    // std::cout<<mat5.getCols()<<std::endl;
    // std::cout<<mat5[pair1]<<std::endl;
    // std::cout<<mat4[pair1]<<std::endl;
    // std::cout<<mat3[pair1]<<std::endl;
    // std::cout<<mat2[pair1]<<std::endl;
    // std::cout<<mat1[pair1]<<std::endl;
    // Matrix<double> mat6(2,2,{1,2,3,4,5});
    // std::pair<int, int> pair2(0,2);
    // std::cout<<mat1[pair2]<<std::endl;
    
    // Matrix<double> mat10(2,3,{1,2,3,4,5,6});
    // Matrix<double> mat7 = mat1*mat10;
    // Matrix<double> mat8 = mat3-mat10;
    // //std::cout<<mat8[pair2]<<std::endl;

    // Matrix<double> mat9 = mat3+mat10;
    double learning_rate = 0.0005;
    int optimizer_steps = 100;
    int seed = 1;
    Matrix<double> xxor(4,2,{0,0,0,1,1,0,1,1});
    Matrix<double> yxor(4,2,{1,0,0,1,0,1,1,0});
    int in_features = 2;
    int hidden_dim = 100;
    int out_features = 2;
    int n_samples = 8;
    for (int i = 0; i<optimizer_steps; i++){
        std::cout<<i<<std::endl;
    Net<double> net(in_features, hidden_dim, out_features, n_samples, seed);
    std::cout<<i<<std::endl;
    Matrix<double> fwd_step = net.forward(xxor);
    std::cout<<i<<std::endl;
    double loss = MSEloss(yxor,fwd_step);
    std::cout<<loss<<std::endl;
    std::cout<<i<<std::endl;
    Matrix<double> gradmat = MSEgrad(yxor,fwd_step);
    std::cout<<i<<std::endl;
    Matrix<double> back_step = net.backward(gradmat);
    std::cout<<i<<std::endl;
    net.optimize(learning_rate);
    std::cout<<i<<std::endl;
    double acc = get_accuracy(yxor,back_step);
    std::cout<<acc<<std::endl;
    }
    return 0;
}
