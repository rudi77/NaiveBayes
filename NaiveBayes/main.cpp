#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <numeric>
#include <sstream>

#include "gnb.h"

using namespace std;

vector<vector<double> > Load_State(string file_name)
{
  ifstream in_state_(file_name.c_str(), ifstream::in);
  vector< vector<double >> state_out;
  string line;


  while (getline(in_state_, line))
  {
    istringstream iss(line);
    vector<double> x_coord;

    string token;
    while (getline(iss, token, ','))
    {
      x_coord.push_back(stod(token));
    }
    state_out.push_back(x_coord);
  }
  return state_out;
}
vector<string> Load_Label(string file_name)
{
  ifstream in_label_(file_name.c_str(), ifstream::in);
  vector< string > label_out;
  string line;
  while (getline(in_label_, line))
  {
    istringstream iss(line);
    string label;
    iss >> label;

    label_out.push_back(label);
  }
  return label_out;

}

void print_dataset(vector<string> labels, vector<vector<double>> features, int chunk_size = 0)
{
  assert(labels.size() == features.size());

  auto func = [](string a, int b) { return a + "\t|\t" + to_string(b); };

  for (auto i=0; i<labels.size(); i++)
  {
    if (chunk_size > 0 && chunk_size == i+1)
    {
      break;
    }

    auto s = std::accumulate(next(features[i].begin()), features[i].end(), to_string(features[i][0]), func);

    cout << labels[i] << "\t|\t" << s << endl;
  }
}

int main() {

  vector< vector<double> > X_train = Load_State("../../../train_states.txt");
  vector< vector<double> > X_test = Load_State("../../../test_states.txt");
  vector< string > Y_train = Load_Label("../../../train_labels.txt");
  vector< string > Y_test = Load_Label("../../../test_labels.txt");

  cout << "X_train number of elements " << X_train.size() << endl;
  cout << "X_train element size " << X_train[0].size() << endl;
  cout << "Y_train number of elements " << Y_train.size() << "\n" << endl;


  print_dataset(Y_test, X_test, 10);

  cout << "\n" << endl;

  auto gnb = GNB();

  gnb.train(X_train, Y_train);

  cout << "X_test number of elements " << X_test.size() << endl;
  cout << "X_test element size " << X_test[0].size() << endl;
  cout << "Y_test number of elements " << Y_test.size() << endl;

  int score = 0;
  for (int i = 0; i < X_test.size(); i++)
  {
    vector<double> coords = X_test[i];
    string predicted = gnb.predict(coords);
    if (predicted.compare(Y_test[i]) == 0)
    {
      score += 1;
    }
  }

  float fraction_correct = float(score) / Y_test.size();
  cout << "You got " << (100 * fraction_correct) << " correct" << endl;

  return 0;
}


