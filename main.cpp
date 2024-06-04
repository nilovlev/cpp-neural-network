#include "except.h"
#include "mnist_test.h"

int main() {
  try {
    neural_network::MnistTest::runAllTests();
  } catch(...) {
    except::react();
  }
  return 0;
}
