#include <Core/array.h>

//===========================================================================

struct SGD{
  double eta;
  double alpha, alpha2;

  arr m, v;
  double t=0.;

  SGD(double _eta=1e-4, double _alpha=.9, double _alpha2=.999) : eta(_eta), alpha(_alpha), alpha2(_alpha2) {}

  void update_plain(arr& x, const arr& g){
    x -= eta * g;
  }

  void update_momentum(arr& x, const arr& g){
    if(!m.N) m = zeros(x.N);
    m = alpha * m - eta * g;
    x += m;
  }

  void update_Nesterov(arr& x, const arr& g){
    if(!m.N) m = zeros(x.N);
    x -= m;
    m = alpha * m - eta * g;
    x += 2.*m;
  }

  void update_Adam(arr& x, const arr& g){
    if(!m.N) m = zeros(x.N);
    if(!v.N) v = zeros(x.N);
    t += 1.;

    m = alpha * m + (1.-alpha) * g;
    v = alpha * v + (1.-alpha2) * (g%g);

    arr mhat = m / (1.-::pow(alpha, t));
    arr vhat = v / (1.-::pow(alpha2, t));
    arr sqrtvhat = sqrt(vhat);

    x -= eta * mhat / (sqrtvhat + 1e-8);
  }

};

void testSGD() {
  uint d=1000, n=10000, k=10;

  //-- generate J
  arr J = zeros(n,d);
  for(uint i=0;i<n;i++){
    double sigm = 1.;
    if(rnd.uni()<.5) sigm=100.;
    for(uint j=0;j<k;j++){
      J(i,rnd(d)) = sigm * rnd.gauss();
    }
  }

  //-- SGD
  //initialize x
  SGD sgd(1e-3);
  arr x = ones(d);
  uint K=32;
  intA select(K);
  arr g(d);
  ofstream fil("z.data");

  for(k=0;k<10000;k++){

    //random mini batch
    rndInteger(select, 0, n-1);

    //compute gradient
    g = 0.;
    double err=0.;
    for(int i:select){
      double Ji_x = scalarProduct(J[i], x);
      g += Ji_x * J[i];
      err += Ji_x * Ji_x;
    }
    g /= double(K);

    //log errors
    double trueErr = 0.5/n * sumOfSqr(J*x);
    double miniErr = 0.5/K * err;
    fil <<k <<' ' <<trueErr <<' ' <<miniErr <<endl;

    if(!(k%100)){
      cout <<k <<" true: " <<trueErr <<" mini: " <<miniErr <<endl;
      gnuplot("plot 'z.data' us 1:2 t 'true', '' us 1:3 t 'mini'", false, true);
    }

    //SGD update
//    sgd.update_plain(x, g);
//    sgd.update_momentum(x, g);
//    sgd.update_Nesterov(x, g);
    sgd.update_Adam(x, g);
  }
}

//===========================================================================

int main(int argc, char *argv[]) {
  rai::initCmdLine(argc,argv);
//  rnd.clockSeed();

  testSGD();

  return 0;
}
