#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

#include "gnb.h"

/**
* Initializes GNB
*/
GNB::GNB() {

}

GNB::~GNB() {}

double mean(vector<double> values)
{
  assert(values.size() > 0);

  return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double stdev(vector<double> values)
{
  assert(values.size() > 0);

  auto avg = mean(values);
  auto t = vector<double>();

  t.resize(values.size());

  transform
  (
    values.begin(),
    values.end(),
    t.begin(),
    [&avg](double v) -> double { return pow(v - avg, 2); }
  );

  auto sum = accumulate(t.begin(), t.end(), 0.0);
  auto variance = sum / static_cast<double>(values.size());
  
  return sqrt(variance);
}

double var(vector<double> values)
{
  assert(values.size() > 0);

  auto avg = mean(values);
  auto t = vector<double>();

  t.resize(values.size());

  transform
  (
    values.begin(),
    values.end(),
    t.begin(),
    [&avg](double v) -> double { return pow(v - avg, 2); }
  );

  auto sum = accumulate(t.begin(), t.end(), 0.0);
  auto variance = sum / static_cast<double>(values.size());

  return variance;
}

double gaussian_prob(double obs, double mu, double sig)
{
  auto num = (obs - mu) * (obs - mu);
  auto denum = 2 * sig*sig;
  auto norm = 1.0 / sqrt(2 * M_PI * sig*sig);

  return norm * exp(-num / denum);
}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
  assert(data.size() > 0 && labels.size() == data.size());

  // Get distinct labels and insert them into a new vector
  unique_copy(labels.begin(), labels.end(), back_inserter(_labels));

  auto label_table = map<string, vector<vector<double>>>();
  auto num_features = data[0].size();

  for (auto ul : _labels)
  {
    label_table[ul] = vector<vector<double>>(num_features);
  }

  // Add feature values
  for (unsigned int i=0; i<data.size(); i++)
  {
    for (unsigned int j=0; j<num_features; j++)
    {
      label_table[labels[i]][j].push_back(data[i][j]);
    }
  }

  // For each label and feature calculate the mean and standard deviation.
  // Therefore, we have two entries mean and stdev for each label.
  for (auto it=label_table.begin(); it != label_table.end(); ++it)
  {
    auto label_values = it->second;

    _means_per_label[it->first] = vector<double>();
    _stdevs_per_label[it->first] = vector<double>();

    for (unsigned int i=0; i<label_values.size(); ++i)
    {
      auto mean = ::mean(label_values[i]);
      auto stdev = ::var(label_values[i]);

      cout << "Label[" << it->first << "], mean: " << mean << ", var: " << stdev << endl;

      _means_per_label[it->first].push_back(mean);
      _stdevs_per_label[it->first].push_back(stdev);
    }

    cout << "\n" << endl;
  }
}

string GNB::predict(vector<double> sample)
{
  assert(_means_per_label.size() == _stdevs_per_label.size());
  
  auto probs = map<string, double>();
  auto probs_sum = 0.0;
  
  // calculate the probability for each label
  for (auto label_name : _labels)
  {
    auto product = 1.0;
    auto mean_values = _means_per_label[label_name];
    auto var_values = _stdevs_per_label[label_name];

    if (mean_values.size() != var_values.size())
    {
      throw runtime_error("mean_values.size() != stdev_values.size()");
    }

    for (auto i=0; i<sample.size(); ++i)
    {
      auto observation = sample[i];
      auto mean = mean_values[i];
      auto var = var_values[i];
      auto likelihood = gaussian_prob(observation, mean, var);

      product *= likelihood;
    }

    probs[label_name] = product;
    probs_sum += product;
  }

  // calc final probs
  for (auto it = probs.begin(); it != probs.end(); ++it)
  {
    it->second /= probs_sum;
  }

  auto max = 0.0;
  string label = "";

  for (auto it = probs.begin(); it != probs.end(); ++it)
  {
    if (max < it->second)
    {
      max = it->second;
      label = it->first;
    }
  }

  return label;
}