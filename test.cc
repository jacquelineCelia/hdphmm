/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./test.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
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
