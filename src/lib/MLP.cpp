//============================================================================
// Name : MLP.cpp
// Author : David Nogueira
// Modified by: Yegor Levonenko
//============================================================================

#include "MLP.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>


//desired call sintax :  MLP({64*64,20,4}, {"sigmoid", "linear"},
MLP_Lib::MLP::MLP(const std::vector<uint64_t> & layers_nodes,
         const std::vector<std::string> & layers_activfuncs,
         bool use_constant_weight_init,
         double constant_weight_init) {
  assert(layers_nodes.size() >= 2);
  assert(layers_activfuncs.size() + 1 == layers_nodes.size());

  CreateMLP(layers_nodes,
            layers_activfuncs,
            use_constant_weight_init,
            constant_weight_init);
};

MLP_Lib::MLP::MLP(const std::string & filename) {
  LoadMLPNetwork(filename);
}

MLP_Lib::MLP::~MLP() {
  m_num_inputs = 0;
  m_num_outputs = 0;
  m_num_hidden_layers = 0;
  m_layers_nodes.clear();
  m_layers.clear();
};

void MLP_Lib::MLP::CreateMLP(const std::vector<uint64_t> & layers_nodes,
                    const std::vector<std::string> & layers_activfuncs,
                    bool use_constant_weight_init,
                    double constant_weight_init) {
  m_layers_nodes = layers_nodes;
  m_num_inputs = m_layers_nodes[0];
  m_num_outputs = m_layers_nodes[m_layers_nodes.size() - 1];
  m_num_hidden_layers = m_layers_nodes.size() - 2;

  for (size_t i = 0; i < m_layers_nodes.size() - 1; i++) {
    m_layers.emplace_back(Layer(m_layers_nodes[i],
                                m_layers_nodes[i + 1],
                                layers_activfuncs[i],
                                use_constant_weight_init,
                                constant_weight_init));
  }
};

void MLP_Lib::MLP::SaveMLPNetwork(const std::string & filename)const {
  FILE * file;
  file = fopen(filename.c_str(), "wb");
  fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
  fwrite(&m_num_outputs, sizeof(m_num_outputs), 1, file);
  fwrite(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
  if (!m_layers_nodes.empty())
    fwrite(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);
  for (size_t i = 0; i < m_layers.size(); i++) {
    m_layers[i].SaveLayer(file);
  }
  fclose(file);
};
void MLP_Lib::MLP::LoadMLPNetwork(const std::string & filename) {
  m_layers_nodes.clear();
  m_layers.clear();

  FILE * file;
  file = fopen(filename.c_str(), "rb");
  fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
  fread(&m_num_outputs, sizeof(m_num_outputs), 1, file);
  fread(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
  m_layers_nodes.resize(m_num_hidden_layers + 2);
  if (!m_layers_nodes.empty())
    fread(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);
  m_layers.resize(m_layers_nodes.size() - 1);
  for (size_t i = 0; i < m_layers.size(); i++) {
    m_layers[i].LoadLayer(file);
  }
  fclose(file);
};

void MLP_Lib::MLP::GetOutput(const std::vector<double> &input,
                    std::vector<double> * output,
                    std::vector<std::vector<double>> * all_layers_activations) const {
  assert(input.size() == m_num_inputs);
  int temp_size;
  if (m_num_hidden_layers == 0)
    temp_size = m_num_outputs;
  else
    temp_size = m_layers_nodes[1];

  std::vector<double> temp_in(m_num_inputs, 0.0);
  std::vector<double> temp_out(temp_size, 0.0);
  temp_in = input;

  for (size_t i = 0; i < m_layers.size(); ++i) {
    if (i > 0) {
      //Store this layer activation
      if (all_layers_activations != nullptr)
        all_layers_activations->emplace_back(std::move(temp_in));

      temp_in.clear();
      temp_in = temp_out;
      temp_out.clear();
      temp_out.resize(m_layers[i].GetOutputSize());
    }
    m_layers[i].GetOutputAfterActivationFunction(temp_in, &temp_out);
  }

  if (temp_out.size() > 1)
    utils::Softmax(&temp_out);
  *output = temp_out;

  //Add last layer activation
  if (all_layers_activations != nullptr)
    all_layers_activations->emplace_back(std::move(temp_in));
}

void MLP_Lib::MLP::GetOutputClass(const std::vector<double> &output, size_t * class_id) const {
  utils::GetIdMaxElement(output, class_id);
}

void MLP_Lib::MLP::UpdateWeights(const std::vector<std::vector<double>> & all_layers_activations,
                        const std::vector<double> &deriv_error,
                        double learning_rate) {

  std::vector<double> temp_deriv_error = deriv_error;
  std::vector<double> deltas{};
  //m_layers.size() equals (m_num_hidden_layers + 1)
  for (int i = m_num_hidden_layers; i >= 0; --i) {
    m_layers[i].UpdateWeights(all_layers_activations[i], temp_deriv_error, learning_rate, &deltas);
    if (i > 0) {
      temp_deriv_error.clear();
      temp_deriv_error = std::move(deltas);
      deltas.clear();
    }
  }
};

void MLP_Lib::MLP::Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
                          double learning_rate,
                          int max_iterations,
                          double min_error_cost,
                          bool output_log) {

  int i = 0;
  double current_iteration_cost_function = 0.0;

  for (i = 0; i < max_iterations; i++) {
    current_iteration_cost_function = 0.0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {

      std::vector<double> predicted_output;
      std::vector< std::vector<double> > all_layers_activations;

      GetOutput(training_sample_with_bias.input_vector(),
                &predicted_output,
                &all_layers_activations);

      const std::vector<double> &  correct_output =
        training_sample_with_bias.output_vector();

      assert(correct_output.size() == predicted_output.size());
      std::vector<double> deriv_error_output(predicted_output.size());

      if (output_log && ((i % (max_iterations / 10)) == 0)) {
        std::stringstream temp_training;
        temp_training << training_sample_with_bias << "\t\t";

        temp_training << "Predicted output: [";
        for (size_t i = 0; i < predicted_output.size(); i++) {
          if (i != 0)
            temp_training << ", ";
          temp_training << predicted_output[i];
        }
        temp_training << "]";
      }

      for (size_t j = 0; j < predicted_output.size(); j++) {
        current_iteration_cost_function +=
          (std::pow)((correct_output[j] - predicted_output[j]), 2);
        deriv_error_output[j] =
          -2 * (correct_output[j] - predicted_output[j]);
      }

      UpdateWeights(all_layers_activations,
                    deriv_error_output,
                    learning_rate);
    }

    if (current_iteration_cost_function < min_error_cost)
      break;
  }
};


size_t MLP_Lib::MLP::GetNumLayers()
{
    return m_layers.size();
}

std::vector<std::vector<double>> MLP_Lib::MLP::GetLayerWeights( size_t layer_i )
{
    std::vector<std::vector<double>> ret_val;
    // check parameters
    if( 0 <= layer_i && layer_i < m_layers.size() )
    {
        Layer current_layer = m_layers[layer_i];
        for( Node & node : current_layer.GetNodesChangeable() )
        {
            ret_val.push_back( node.GetWeights() );
        }
        return ret_val;
    }
    else
        throw new std::logic_error("Incorrect layer number in GetLayerWeights call");

}

void MLP_Lib::MLP::SetLayerWeights( size_t layer_i, std::vector<std::vector<double>> & weights )
{
    // check parameters
    if( 0 <= layer_i && layer_i < m_layers.size() )
    {
        m_layers[layer_i].SetWeights( weights );
    }
    else
        throw new std::logic_error("Incorrect layer number in SetLayerWeights call");
}


