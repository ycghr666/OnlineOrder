param a;
param b;
param c;
param d;
param e;

var x {1..5} >= 0;

maximize z: 7 - 3*x[4] - a*x[5];

subject to Row1:
    x[3] + 4*x[4] + b*x[5] = c;

subject to Row2:
    x[1] - 7*x[4] + d*x[5] = 2;

subject to Row3:
    x[2] + e*x[4] - 2*x[5] = 4;
