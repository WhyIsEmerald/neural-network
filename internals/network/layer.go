package network

import (
	"math/rand"

	"github.com/whyisemerald/neural_network/internals/matrix"
)

type Activation func(float64) float64

type Layer struct {
	Weights              *matrix.Matrix
	Biases               *matrix.Matrix
	activation           Activation
	activationDerivative Activation
	numNeurons           int
	numInputs            int

	// Pre-allocated matrices
	Output            *matrix.Matrix
	Deltas            *matrix.Matrix
	Inputs            *matrix.Matrix
	rawOutput         *matrix.Matrix
	weightGradients   *matrix.Matrix
	biasGradients     *matrix.Matrix
	errorMatrix       *matrix.Matrix
	derivativeMatrix  *matrix.Matrix
}

func NewLayer(numNeurons, numInputs int, activation, activationDerivative Activation) *Layer {
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
		Weights:              weights,
		Biases:               biases,
		activation:           activation,
		activationDerivative: activationDerivative,
		numNeurons:           numNeurons,
		numInputs:            numInputs,

		// Pre-allocate matrices. Assumes a batch size of 1.
		Output:           matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
		Deltas:           matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
		rawOutput:        matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
		weightGradients:  matrix.NewMatrix(numInputs, numNeurons, make([]float64, numInputs*numNeurons)),
		biasGradients:    matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
		errorMatrix:      matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
		derivativeMatrix: matrix.NewMatrix(1, numNeurons, make([]float64, numNeurons)),
	}
}

func (l *Layer) Forward(inputs *matrix.Matrix) *matrix.Matrix {
	l.Inputs = inputs

	// Calculations use pre-allocated matrices. Assumes a consistent batch size.
	matrix.DotProduct(inputs, l.Weights, l.rawOutput)
	matrix.Add(l.rawOutput, l.Biases, l.rawOutput)
	matrix.ApplyFunction(l.rawOutput, l.activation, l.Output)

	return l.Output
}

func (l *Layer) Update(learningRate float64) {
	inputsT := matrix.Transpose(l.Inputs)
	matrix.DotProduct(inputsT, l.Deltas, l.weightGradients)

	matrix.MultiplyScalar(l.weightGradients, learningRate, l.weightGradients)

	matrix.Subtract(l.Weights, l.weightGradients, l.Weights)

	matrix.MultiplyScalar(l.Deltas, learningRate, l.biasGradients)
	matrix.Subtract(l.Biases, l.biasGradients, l.Biases)
}
