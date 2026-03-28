#pragma once
#include "types.h"

struct Lattice {
    int L;
    int V;
    int ndof;
    int V_half;   // V/2 — number of even (or odd) sites

    // Even-odd site index maps
    std::vector<int> even_sites;  // even_sites[i] = global site index of i-th even site
    std::vector<int> odd_sites;   // odd_sites[i]  = global site index of i-th odd site
    std::vector<int> eo_index;    // eo_index[s]   = index within even or odd array

    Lattice(int L_) : L(L_), V(L_*L_), ndof(2*L_*L_), V_half(L_*L_/2) {
        init_eo();
    }

    int idx(int x, int y) const { return ((x%L+L)%L) + L*((y%L+L)%L); }

    int xp(int s) const { int x=s%L, y=s/L; return idx(x+1,y); }
    int xm(int s) const { int x=s%L, y=s/L; return idx(x-1,y); }
    int yp(int s) const { int x=s%L, y=s/L; return idx(x,y+1); }
    int ym(int s) const { int x=s%L, y=s/L; return idx(x,y-1); }

    int parity(int s) const { return ((s % L) + (s / L)) % 2; }

    void init_eo() {
        even_sites.reserve(V_half);
        odd_sites.reserve(V_half);
        eo_index.resize(V);
        for (int s = 0; s < V; s++) {
            if (parity(s) == 0) {
                eo_index[s] = (int)even_sites.size();
                even_sites.push_back(s);
            } else {
                eo_index[s] = (int)odd_sites.size();
                odd_sites.push_back(s);
            }
        }
    }
};
