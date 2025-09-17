package network

import "github.com/whyisemerald/neural_network/internals/math"

type Network struct {
	Layers []*Layer
}

func NewNetwork(layerSizes []int) *Network {
	layers := make([]*Layer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		layers[i] = NewLayer(layerSizes[i+1], layerSizes[i])
	}
	return &Network{Layers: layers}
}

func (n *Network) Forward(inputs []float64) []float64 {
	currentInputs := inputs
	if len(inputs) != n.Layers[0].numInputs {
		panic("Input size does not match the number of inputs of the first layer")
	}
	for _, layer := range n.Layers {
		outputs := make([]float64, len(layer.Neurons))
		for i, neuron := range layer.Neurons {
			outputs[i] = neuron.resolveOutput(currentInputs)
		}
		currentInputs = outputs
	}
	return currentInputs
}

func (n *Network) Backward(expected []float64) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		if i == len(n.Layers)-1 {
			for j, neuron := range layer.Neurons {
				neuron.delta = (expected[j] - neuron.output) * math.SigmoidDerivative(neuron.output)
			}
		} else {
			for j, neuron := range layer.Neurons {
				error := 0.0
				for _, nextNeuron := range n.Layers[i+1].Neurons {
					error += nextNeuron.weights[j] * nextNeuron.delta
				}
				neuron.delta = error * math.SigmoidDerivative(neuron.output)
			}
		}
	}
}

func (n *Network) UpdateWeights(learningRate float64) {
	for _, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			for i, input := range neuron.inputs {
				neuron.weights[i] += learningRate * neuron.delta * input
			}
			neuron.bias += learningRate * neuron.delta
		}
	}
}

func (n *Network) Train(inputs, expected []float64, learningRate float64) {
	n.Forward(inputs)
	n.Backward(expected)
	n.UpdateWeights(learningRate)
}

func (n *Network) TrainLoop(input, expected [][]float64, learningRate float64, epoch int) {
	for i := 0; i < epoch; i++ {
		for j, in := range input {
			n.Train(in, expected[j], learningRate)
		}
	}
}
