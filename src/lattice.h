#pragma once
#include "types.h"
#include <vector>

struct Lattice {
    int L;
    int V;
    int ndof;

    // Even-odd decomposition: even sites have (x+y)%2==0
    int V2;                           // V/2 = number of sites per parity
    int ndof2;                        // ndof/2 = DOF per parity
    std::vector<int> eo_map;          // full site → half-site index within parity
    std::vector<int> even_sites;      // half-index → full site (even)
    std::vector<int> odd_sites;       // half-index → full site (odd)

    Lattice(int L_) : L(L_), V(L_*L_), ndof(2*L_*L_), V2(L_*L_/2), ndof2(L_*L_)
    {
        eo_map.resize(V);
        even_sites.reserve(V2);
        odd_sites.reserve(V2);
        for (int s = 0; s < V; s++) {
            int x = s % L, y = s / L;
            if ((x + y) % 2 == 0) {
                eo_map[s] = (int)even_sites.size();
                even_sites.push_back(s);
            } else {
                eo_map[s] = (int)odd_sites.size();
                odd_sites.push_back(s);
            }
        }
    }

    int idx(int x, int y) const { return ((x%L+L)%L) + L*((y%L+L)%L); }

    int xp(int s) const { int x=s%L, y=s/L; return idx(x+1,y); }
    int xm(int s) const { int x=s%L, y=s/L; return idx(x-1,y); }
    int yp(int s) const { int x=s%L, y=s/L; return idx(x,y+1); }
    int ym(int s) const { int x=s%L, y=s/L; return idx(x,y-1); }

    bool is_even(int s) const { int x = s % L, y = s / L; return (x + y) % 2 == 0; }
};
