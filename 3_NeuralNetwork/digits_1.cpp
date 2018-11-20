#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
#include <fstream>
#include <stdint.h>

class NeuralNetwork {
  /* A simple neural network. The code is written with the help of the 
     book "Neural networks and deep learning" by Michael Nielsen
     ================================================================
     neuralnetworksanddeeplearning.com
     ================================================================
     n contains the number of layers (input layer, output 
     layer and all hidden layers), vector layers contains the number 
     of nodes in each layer. Outputs vector contains vectors of outputs of 
     each layer. Vector deltas contains the errors for each layer. 
     Vector zs contains the weighted inputs for each layer.  Vector dCdb 
     contains the gradient of the cost function wrt biases. Vector dCdw
     contains matrices of derivatives of the cost function wrt elements 
     of the weight matrices. eta is the learning rate.
     Method feedforward returns an output of the network for a given input.
     Method sigma contains the activation function (sigmoid in this case). 
     Its derivative is in method sigma_prime. Method dCda returns the vector
     corresponding to the gradient of the cost function wrt output layer 
     activations. compute_errors computes the errors using the backpropagation
     method. Method compute_gradient computes the derivatives dCdb and dCdw. 
     Method sgd_step calculates the new iteration of the weights and biases
     using the stochastic gradient method. The input minibatch of the samples 
     is used. The output of the network after learning can be checked by 
     using method control. 
     The length of vectors layers and outputs is n. The length of vectors
     biases, weights, deltas, zs, dCdb, dCdw is n-1. */
private:
  unsigned int n;
  std::vector<std::size_t> layers;
  std::vector<std::vector<double> > outputs;
  std::vector<std::vector<double> > biases;
  std::vector<boost::numeric::ublas::matrix<double> > weights;
  std::vector<std::vector<double> > deltas;
  std::vector<std::vector<double> > zs;
  std::vector<std::vector<double> > dCdb;
  std::vector<boost::numeric::ublas::matrix<double> > dCdw;
  double eta;
public:
  NeuralNetwork();
  NeuralNetwork(std::size_t);
  NeuralNetwork(std::size_t, std::vector<std::size_t>);
  std::vector<double> feedforward(std::vector<double>);
  double sigma(double);
  double sigma_prime(double);
  std::vector<double> dCda(std::vector<double>, std::vector<double>);
  void compute_errors(std::vector<double>,std::vector<double>);
  void compute_gradient();
  void sgd_step(std::vector<std::vector<double> >,std::vector<std::vector<double> >);
  std::vector<double> control(std::vector<double>);
};

NeuralNetwork::NeuralNetwork() {
  n = 3;
  eta = 4;
  for(std::size_t i=0; i<n; ++i)
    layers.push_back(n);
}

NeuralNetwork::NeuralNetwork(std::size_t _n_, std::vector<std::size_t> _layers_) {
  if(_layers_.size()==_n_) {
    n = _n_;
    layers = _layers_;
    eta = 4;
    /* Allocate the output vector of the input 
       layer (aka input of the network). */
    std::vector<double> a;
    for(std::size_t j=0; j<_layers_[0]; ++j) {
      a.push_back(0.0);
    }
    outputs.push_back(a);
    std::random_device rd;
    std::mt19937 gen(rd());
    /* Notice how the loop starts with i=1. */
    for(std::size_t i=1; i<_n_; ++i) {
      /* Creates a n1 x n2 matrix filled with normally distributed
	 numbers of mean 0 and standard deviation 1/sqrt(m) where m
	 is the number of inputs to the particular neuron (in this 
	 case all neurons are connected to all neurons in the next
	 layer and therefore this number is const for all the 
	 neurons in the current layer). */
      std::normal_distribution<double> dis(0, 1.0/sqrt(_layers_[i-1]));
      std::size_t n1 = _layers_[i-1], n2 = _layers_[i];
      boost::numeric::ublas::matrix<double> M(n1,n2);
      boost::numeric::ublas::matrix<double> dM(n1,n2);
      for(std::size_t j=0; j<n1; ++j) {
	for(std::size_t k=0; k<n2; ++k) {
	  M(j,k) = dis(gen);
	  dM(j,k) = 0.0;
	}
      }
      weights.push_back(M);
      dCdw.push_back(dM);
      std::vector<double> v;
      std::vector<double> w;
      for(std::size_t j=0; j<_layers_[i]; ++j) {
	v.push_back(dis(gen));
	w.push_back(0.0);
      }
      biases.push_back(v);
      dCdb.push_back(w);
      outputs.push_back(w);
      deltas.push_back(w);
      zs.push_back(w);
    }
  }
  else
    throw "Neural network was not initialized correctly!";
}

std::vector<double> NeuralNetwork::feedforward(std::vector<double> input) {
  std::vector<double> a_prev;
  std::vector<double> a;
  a_prev = input;
  for(std::size_t i=1; i<layers.size(); ++i) {
    a = outputs[i];
    for(std::size_t j=0; j<layers[i]; ++j) {
      double sum = 0.0;
      for(std::size_t k=0; k<layers[i-1]; ++k)
	sum += weights[i-1](k,j)*a_prev[k];
      zs[i-1][j] = sum + biases[i-1][j];
      a[j] = sigma(zs[i-1][j]);
    }
    a_prev = a;
    outputs[i] = a;
  }
  return a;
}

double NeuralNetwork::sigma(double x) {
  return 1.0/(1.0+exp(-x));
}

double NeuralNetwork::sigma_prime(double x) {
  return sigma(x)*(1.0-sigma(x));
}

std::vector<double> NeuralNetwork::dCda(std::vector<double> a, std::vector<double> a_expected) {
    if(a.size()!=a_expected.size())
      throw "The output vector and the vector of expected results are of different sizes!";
    else{
      std::vector<double> r;
      for(std::size_t i=0; i<a.size(); ++i)
	r.push_back(a[i]-a_expected[i]);
      return r;
    }
}

void NeuralNetwork::compute_errors(std::vector<double> network_output,
				   std::vector<double> a_expected) {
  /* Compute delta for the output layer. */
  std::vector<double> dC_da = dCda(network_output, a_expected);
  for(std::size_t i=0; i<deltas[n-2].size(); ++i) {
    deltas[n-2][i] = dC_da[i]*sigma_prime(zs[n-2][i]);
  }
  /* Compute the rest of the deltas using recurrent expression. */
  for(std::size_t ll=0; ll<=n-3; ++ll) {
    /* Notice how you can't use 
       for(std::size_t l=n-3; l>=0; --l) 
       because std::size_t cannot be negative and will return a 
       very big number after it passes zero therefore not terminating
       the loop and going out of boundaries of the vector indices. */
    int l = n-3-ll;
    for(std::size_t j=0; j<deltas[l].size(); ++j) {
      double sum = 0.0;
      for(std::size_t k=0; k<deltas[l+1].size(); ++k) {
	sum += deltas[l+1][k]*weights[l+1](j,k);
      }
      deltas[l][j] = sum*sigma_prime(zs[l][j]);
    }
  }
}

void NeuralNetwork::compute_gradient() {
  for(std::size_t l=0; l<n-1; ++l) {
    for(std::size_t j=0; j<layers[l+1]; ++j) {
      for(std::size_t k=0; k<layers[l]; ++k) {
	dCdw[l](k,j) = deltas[l][j]*outputs[l][k];
      }
      dCdb[l][j] = deltas[l][j];
    }
  }
}

void NeuralNetwork::sgd_step(std::vector<std::vector<double> > minibatch,
			     std::vector<std::vector<double> > answers) {
  /* The following two structures will accumulate
     the gradients dCdb and dCdw from each training 
     example x. */
  std::vector<std::vector<double> > dCdb_;
  std::vector<boost::numeric::ublas::matrix<double> > dCdw_;
  /* We need to initialize them too. */
  for(std::size_t ii=1; ii<n; ++ii) {
    std::size_t n1 = layers[ii-1], n2 = layers[ii];
    boost::numeric::ublas::matrix<double> dM(n1,n2);
    for(std::size_t j=0; j<n1; ++j) {
      for(std::size_t k=0; k<n2; ++k) {
	dM(j,k) = 0.0;
      }
    }
    dCdw_.push_back(dM);
    std::vector<double> w;
    for(std::size_t j=0; j<layers[ii]; ++j) {
      w.push_back(0.0);
    }
    dCdb_.push_back(w);
  }

  std::size_t m = minibatch.size();
  for(std::size_t i=0; i<m; ++i) {
    std::vector<double> x = minibatch[i];
    std::vector<double> y = answers[i];
    
    if(x.size()!=layers[0])
      throw "The input vector and the input layer are of different sizes!";
    for(std::size_t j=0; j<layers[0]; ++j) {
      outputs[0][j] = sigma(x[j]);
    }
    for(std::size_t l=1; l<n; ++l) {
      for(std::size_t j=0; j<layers[l]; ++j) {
	dCdb[l-1][j] = 0.0;
	for(std::size_t k=0; k<layers[l-1]; ++k) {
	  dCdw[l-1](k,j) = 0.0;
	}
      }
    }
    
    std::vector<double> network_output = feedforward(outputs[0]);
    compute_errors(network_output,y);
    compute_gradient();
    for(std::size_t l=1; l<n; ++l) {
      for(std::size_t j=0; j<layers[l]; ++j) {
	dCdb_[l-1][j] += dCdb[l-1][j];
	for(std::size_t k=0; k<layers[l-1]; ++k) {
	  dCdw_[l-1](k,j) += dCdw[l-1](k,j);
	}
      }
    }
  }
  for(std::size_t l=1; l<n; ++l) {
    for(std::size_t j=0; j<layers[l]; ++j) {
      biases[l-1][j] += -eta/m*dCdb_[l-1][j];
      for(std::size_t k=0; k<layers[l-1]; ++k) {
	weights[l-1](k,j) += -eta/m*dCdw_[l-1](k,j);
      }
    }
  }
}

std::vector<double> NeuralNetwork::control(std::vector<double> x) {
  /* This routine is used to compute 
     the output of the network during 
     actual production run with test data. */
  for(std::size_t j=0; j<layers[0]; ++j) {
    outputs[0][j] = sigma(x[j]);
  }
  std::vector<double> result = feedforward(outputs[0]);
  return result;
}

uint32_t inverse_int(uint32_t i){
  /* This subroutine helps to convert
     the high-endian data from the MNIST
     set to the low-endian format used on 
     standard PCs. 4 byte integer number i
     is input and its bytes are swapped
     efficiently in order to extract the 
     correct information (an integer number). */
  uint32_t i1,i2,i3,i4;
  i1 = i & 255;
  i2 = (i>>8) & 255;
  i3 = (i>>16) & 255;
  i4 = (i>>24) & 255;
  return (i1<<24) + (i2<<16) + (i3<<8) + i4;
}

std::vector<std::vector<int> > extract_images(std::string images_file,
					      std::string labels_file) {
  /* This routine takes the data files containing the
     MNIST data set that can be downloaded from 
     http://yann.lecun.com/exdb/mnist/
     and returns the c++ vector containing the images as 
     elements. Images are represented by c++ vectors of
     ints of length 785. First 784 components of each
     vector contain integers (values in range [0,255]) 
     that code the greyscale (0=white, 255=black) color of each of the 
     28x28 pixels of the image of the handwritten digit. 
     The last, 785th element contains an integer value from 0 to 9
     corresponding to what digit is depicted on the image. 
     One needs to provide both the image file and the labels 
     (i.e., key) file simultaneously. Files are assumed to be 
     decompressed out of the gz archive. */
  std::vector<std::vector<int> > images;
  std::ifstream data;
  data.open(images_file,std::ios::binary);
  uint32_t n;
  std::size_t 
    magic_number = 0,
    n_images = 0,
    n_rows = 0,
    n_columns = 0;
  if(data.is_open()) {
    data.read((char *)&n,sizeof(uint32_t));
    magic_number = inverse_int(n);
    data.read((char *)&n,sizeof(uint32_t));
    n_images = inverse_int(n);
    data.read((char *)&n,sizeof(uint32_t));
    n_rows = inverse_int(n);
    data.read((char *)&n,sizeof(uint32_t));
    n_columns = inverse_int(n);
    std::cout << "magic number is " << magic_number << std::endl;
    std::cout << "number of images is " << n_images << std::endl;
    std::cout << "number of rows is " << n_rows << std::endl;
    std::cout << "number of columns is " << n_columns << std::endl;

    /* Actually record the images into the c++ vectors. */
    uint8_t m;
    for(std::size_t k = 0; k<n_images; ++k) {
      std::vector<int> image;
      for(std::size_t i = 0; i<n_rows; ++i) {
	for(std::size_t j = 0; j<n_columns; ++j) {
	  data.read((char *)&m,sizeof(uint8_t));
	  image.push_back((int)m);
	}
      }
      images.push_back(image);
    }
  }
  else {
    throw "Error opening images file!";
  }
  data.close();

  /* Read the test labels file. */
  data.open(labels_file,std::ios::binary);
  if(data.is_open()) {
    data.read((char *)&n,sizeof(uint32_t));
    magic_number = inverse_int(n);
    data.read((char *)&n,sizeof(uint32_t));
    n_images = inverse_int(n);

    std::cout << "magic number is " << magic_number << std::endl;
    std::cout << "number of images is " << n_images << std::endl;
    /* Actually record the images into the c++ vectors. */
    uint8_t m;
    for(std::size_t k = 0; k<n_images; ++k) {
      std::vector<int> image;
      data.read((char *)&m,sizeof(uint8_t));
      images[k].push_back((int)m);
    }
  }
  else {
    throw "Error opening labels file!";
  }
  data.close();
  return images;
}

std::vector<double> vectorize(int x) {
  /* This subroutine is used to convert the
     single integer value x that denotes the digit 
     on the image into a vector which contains 
     10 elements, 9 of which are zero except an 
     element with index x that contains 1.0.
     This routine is necessary in order to compare 
     the output of the network (10d vector) 
     to the desired outcome. */
  std::vector<double> r(10,0.0);
  r[x] = 1.0;
  return r;
}

int main() {
  /* This program uses the NeuralNetwork class to classificate
     handwritten digits from the MNIST data set. This implementation 
     of the network does not use any optimizations besides rescaling
     of the gaussian distribution for the initialization of the weights
     and biases. The corresponding error rate for
     eta = 4
     mini-batch size = 10
     number of epochs = 10
     is approximately 6.6%. */
  std::size_t N = 10; //number of epochs
  std::size_t n = 3;
  std::size_t counter = 0;
  /* The nature of the input data (images of 28X28=784 pixels) implies
     the number of the nodes in the input layer. Since we would like 
     to answer whether the input image presents a number from 0 to 9 
     there are 10 nodes in the output layer. The output is number x
     if the xth component of the output vector is the largest. */
  int layers_[] = {784,100,10};
  std::vector<std::size_t> layers(layers_, layers_+sizeof(layers_)/sizeof(int));
  NeuralNetwork nn(n,layers);
  /* Use your local path to the unpacked data files. */
  std::string t1 = "./mnist_data/train-images-idx3-ubyte";
  std::string t2 = "./mnist_data/train-labels-idx1-ubyte";
  std::string c1 = "./mnist_data/t10k-images-idx3-ubyte";
  std::string c2 = "./mnist_data/t10k-labels-idx1-ubyte";
  try{
    std::vector<std::vector<int> > train_images = extract_images(t1,t2);
    std::vector<std::vector<int> > test_images = extract_images(c1,c2);

    /* Initiate training phase. */
    for(std::size_t epoch=1; epoch<N; ++epoch) {
      counter = 0;
      std::vector<std::size_t> deck;
      /* The train set of MNIST contains 60000 images. I create 
	 a vector of labels (1 for each train image) and in the 
	 beginning of each epoch i shuffle their order with the help 
	 of c++ method random_shuffle. Mini-batches are taken from 
	 the shuffled set one by one till the training set is 
	 completely exhausted. */
      for(std::size_t i=0; i<60000; ++i)
	deck.push_back(i);
      std::random_shuffle(deck.begin(), deck.end());
      for(std::size_t m=1; m<=6000; ++m) {
	std::vector<std::vector<double> > mb;
	std::vector<std::vector<double> > answers;
	for(std::size_t i=1; i<=10; ++i) {
	  std::vector<double> mb_(784);
	  std::vector<double> answers_(10);
	  for(std::size_t j=0; j<784; ++j) {
	    mb_[j] = train_images[deck[counter]][j]/127.5-1.0;
	  }
	  answers_ = vectorize(train_images[deck[counter]][784]);
	  ++counter;
	  mb.push_back(mb_);
	  answers.push_back(answers_);
	}
	nn.sgd_step(mb,answers);
      }
      std::cout << "Epoch " << epoch << " complete..." << std::endl;
    }

    /* Initiate control phase. */
    counter = 0;
    int success = 0;
    /* The test set from MNIST set contains 10000 images. */
    for(std::size_t m=1; m<=1e4; ++m) {
      std::vector<double> mb(784);
      std::vector<double> answers(10);
      for(std::size_t j=0; j<784; ++j) {
	mb[j] = test_images[counter][j]/127.5-1.0;
      }
      answers = vectorize(test_images[counter][784]);
      std::vector<double> outcome = nn.control(mb);
      double max_el = 0.0;
      std::size_t max_in = 0;
      /* Read out which digit the network thinks is
	 depicted on the image. */
      for(std::size_t i=0; i<10; ++i) {
	if(outcome[i]>max_el) {
	  max_el = outcome[i];
	  max_in = i;
	}
      }
      /* If the network is correct, increase the 
	 number of successful experiments by 1. */
      if(max_in==test_images[counter][784])
	++success;
      ++counter;
    }
    std::cout << "Error rate is " << 1.0-(double)success/1e4 << std::endl;
  }
  catch (const char* msg) {
    std::cerr << msg << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}
