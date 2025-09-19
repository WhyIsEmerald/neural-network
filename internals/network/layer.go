package network

import (
	"math/rand"
	"time"
)

type Layer struct {
	Neurons    []*neuron
	numNeurons int
	numInputs  int
}

func NewLayer(numNeurons, numInputs int) *Layer {
	neurons := make([]*neuron, numNeurons)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < numNeurons; i++ {
		weights := make([]float64, numInputs)

		for j := range weights {
			weights[j] = rand.Float64() - 0.5
		}
		neurons[i] = newNeuron(weights, rand.Float64()-0.5)
	}
	return &Layer{Neurons: neurons, numNeurons: numNeurons, numInputs: numInputs}
}

func (L *Layer) resolveOutputs(inputs []float64) []float64 {
	outputs := make([]float64, len(L.Neurons))
	for i, neuron := range L.Neurons {
		outputs[i] = neuron.resolveOutput(inputs)
	}
	return outputs
}

func (L *Layer) update(learningRate float64) {
	for _, neuron := range L.Neurons {
		for i, input := range neuron.inputs {
			neuron.weights[i] += learningRate * neuron.delta * input
		}
		neuron.bias += learningRate * neuron.delta
	}
}
