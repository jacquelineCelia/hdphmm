#include <iostream>
#include "uniformer.h"

using namespace std;

int main() {
    Uniformer *_uniform = new Uniformer(500000);
    cout << _uniform -> GetSample() << endl;
    cout << _uniform -> GetSample() << endl;
    delete _uniform;
    return 0;
}
