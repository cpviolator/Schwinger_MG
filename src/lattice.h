#pragma once
#include "types.h"

struct Lattice {
    int L;
    int V;
    int ndof;

    Lattice(int L_) : L(L_), V(L_*L_), ndof(2*L_*L_) {}

    int idx(int x, int y) const { return ((x%L+L)%L) + L*((y%L+L)%L); }

    int xp(int s) const { int x=s%L, y=s/L; return idx(x+1,y); }
    int xm(int s) const { int x=s%L, y=s/L; return idx(x-1,y); }
    int yp(int s) const { int x=s%L, y=s/L; return idx(x,y+1); }
    int ym(int s) const { int x=s%L, y=s/L; return idx(x,y-1); }
};
