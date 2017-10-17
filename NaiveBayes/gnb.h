#pragma once

#include <iostream>
#include <vector>
#include <map>

using namespace std;

class GNB {
public:
  GNB();

  virtual ~GNB();

  void train(vector<vector<double> > data, vector<string>  labels);

  string predict(vector<double>);
private:
  vector<string> _labels;
  map<string, vector<double>> _means_per_label;
  map<string, vector<double>> _stdevs_per_label;
};




