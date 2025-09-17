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
