#pragma once
#include "types.h"
#include <random>

cx dot(const Vec& a, const Vec& b);
double norm(const Vec& v);
void axpy(cx a, const Vec& x, Vec& y);
Vec add(const Vec& a, const Vec& b, cx alpha=1.0, cx beta=1.0);
void scale(Vec& v, cx a);
Vec zeros(int n);
Vec random_vec(int n, std::mt19937& rng);
