package network

import (
	"math"
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
	n.output = sigmoid(output)
	return n.output
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func softmax(x []float64) []float64 {
	output := make([]float64, len(x))
	sum := 0.0
	for i := range x {
		output[i] = math.Exp(x[i])
		sum += output[i]
	}
	for i := range output {
		output[i] /= sum
	}
	return output
}
