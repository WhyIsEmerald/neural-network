package network

import (
	"github.com/whyisemerald/neural_network/internals/math"
)

type neuron struct {
	inputs  []float64
	weights []float64
	bias    float64
	output  float64
}

func newNeuron(weights []float64, bias float64) *neuron {
	return &neuron{
		weights: weights,
		bias:    bias,
	}
}

func (n *neuron) replaceWeights(newWeights []float64) {
	n.weights = newWeights
}

func (n *neuron) editWeight(newWeight float64, i int) {
	n.weights[i] = newWeight
}

func (n *neuron) editBias(newBias float64) {
	n.bias = newBias
}

func (n *neuron) resolveOutput(inputs []float64) float64 {
	n.inputs = inputs
	output := 0.0
	for i := range n.inputs {
		output += n.inputs[i] * n.weights[i]
	}
	output += n.bias
	n.output = math.Sigmoid(output)
	return n.output
}
