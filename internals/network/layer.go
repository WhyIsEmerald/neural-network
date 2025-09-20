package network

import (
	"math/rand"

	"github.com/whyisemerald/neural_network/internals/math"
	"github.com/whyisemerald/neural_network/internals/matrix"
)

type Layer struct {
	Weights    *matrix.Matrix
	Biases     *matrix.Matrix
	Output     *matrix.Matrix
	Deltas     *matrix.Matrix
	Inputs     *matrix.Matrix
	numNeurons int
	numInputs  int
}

func NewLayer(numNeurons, numInputs int) *Layer {
	weightsData := make([]float64, numInputs*numNeurons)
	for i := range weightsData {
		weightsData[i] = rand.Float64() - 0.5
	}
	weights := matrix.NewMatrix(numInputs, numNeurons, weightsData)

	biasesData := make([]float64, numNeurons)
	for i := range biasesData {
		biasesData[i] = rand.Float64() - 0.5
	}
	biases := matrix.NewMatrix(1, numNeurons, biasesData)

	return &Layer{
		Weights:    weights,
		Biases:     biases,
		numNeurons: numNeurons,
		numInputs:  numInputs,
	}
}

func (l *Layer) Forward(inputs *matrix.Matrix) *matrix.Matrix {
	l.Inputs = inputs

	rawOutput := matrix.NewMatrix(inputs.Rows, l.Weights.Cols, make([]float64, inputs.Rows*l.Weights.Cols))
	matrix.DotProduct(inputs, l.Weights, rawOutput)
	matrix.Add(rawOutput, l.Biases, rawOutput)

	l.Output = matrix.NewMatrix(rawOutput.Rows, rawOutput.Cols, make([]float64, len(rawOutput.Data)))
	matrix.ApplyFunction(rawOutput, math.Sigmoid, l.Output)

	return l.Output
}

func (l *Layer) Update(learningRate float64) {
	inputsT := matrix.Transpose(l.Inputs)
	weightGradients := matrix.NewMatrix(inputsT.Rows, l.Deltas.Cols, make([]float64, inputsT.Rows*l.Deltas.Cols))
	matrix.DotProduct(inputsT, l.Deltas, weightGradients)

	matrix.MultiplyScalar(weightGradients, learningRate, weightGradients)

	matrix.Subtract(l.Weights, weightGradients, l.Weights)

	biasGradients := matrix.NewMatrix(l.Deltas.Rows, l.Deltas.Cols, make([]float64, len(l.Deltas.Data)))
	matrix.MultiplyScalar(l.Deltas, learningRate, biasGradients)
	matrix.Subtract(l.Biases, biasGradients, l.Biases)
}
