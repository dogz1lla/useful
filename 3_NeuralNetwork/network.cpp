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
  eta = 0.5;
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
  for(std::size_t j=0; j<layers[0]; ++j) {
    outputs[0][j] = sigma(x[j]);
  }
  std::vector<double> result = feedforward(outputs[0]);
  return result;
}

int main() {
  /* As a test one can apply the described                                                            
  network as a classificator between unit                                                            
  vectors [1,0] and [0,1]. */ 
  std::size_t n = 3;
  int layers_[] = {2,3,2};
  std::vector<std::size_t> layers(layers_, layers_+sizeof(layers_)/sizeof(int));
  NeuralNetwork nn(n,layers);
  srand(time(NULL));
  for(std::size_t m=0; m<10000; m++) {
    std::vector<std::vector<double> > mb;
    std::vector<std::vector<double> > answers;
    for(std::size_t i=0; i<10; ++i) {
      std::vector<double> mb_(2);
      std::vector<double> an_(2);
      if(rand()%2) {
	mb_[0] = 1.0;
	mb_[1] = 0.0;
	an_[0] = 1.0;
	an_[1] = 0.0;
      } else {
	mb_[0] = 0.0;
	mb_[1] = 1.0;
	an_[0] = 0.0;
	an_[1] = 1.0;
      }
      mb.push_back(mb_);
      answers.push_back(an_);
    }
    nn.sgd_step(mb,answers);
  }

  for(std::size_t i=0; i<2; ++i) {
    std::vector<double> mb_(2);
    if(rand()%2) {
      mb_[0] = 1.0;
      mb_[1] = 0.0;
    } else {
      mb_[0] = 0.0;
      mb_[1] = 1.0;
    }
    std::cout << "[" << mb_[0] << ", " << mb_[1] << "]" << std::endl;
    std::vector<double> an_ = nn.control(mb_);
    std::cout << "[" << an_[0] << ", " << an_[1] << "]" << std::endl;
  }
  return 0;
}
