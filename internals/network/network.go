package network

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/whyisemerald/neural_network/internals/math"
	"github.com/whyisemerald/neural_network/internals/matrix"
)

type Network struct {
	Layers []*Layer
}
type NetworkData struct {
	Weights [][][]float64
	Biases  [][]float64
}

func NewNetwork(layerSizes []int) *Network {
	layers := make([]*Layer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		layers[i] = NewLayer(layerSizes[i+1], layerSizes[i])
	}
	return &Network{Layers: layers}
}

func (n *Network) Forward(inputs *[]float64) []float64 {
	if len(*inputs) != n.Layers[0].numInputs {
		panic("Input size does not match the number of inputs of the first layer")
	}

	currentInputsMatrix := matrix.NewMatrix(1, len(*inputs), *inputs)

	for _, layer := range n.Layers {
		currentInputsMatrix = layer.Forward(currentInputsMatrix)
	}

	return currentInputsMatrix.Data
}

func (n *Network) Backward(expected *[]float64) {
	expectedMatrix := matrix.NewMatrix(1, len(*expected), *expected)

	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		if i == len(n.Layers)-1 {
			errorMatrix := matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.Subtract(layer.Output, expectedMatrix, errorMatrix)

			derivativeMatrix := matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.ApplyFunction(layer.Output, math.SigmoidDerivative, derivativeMatrix)

			layer.Deltas = matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.MultiplyElementWise(errorMatrix, derivativeMatrix, layer.Deltas)

		} else {
			nextLayer := n.Layers[i+1]

			weightsT := matrix.Transpose(nextLayer.Weights)
			errorMatrix := matrix.NewMatrix(nextLayer.Deltas.Rows, weightsT.Cols, make([]float64, nextLayer.Deltas.Rows*weightsT.Cols))
			matrix.DotProduct(nextLayer.Deltas, weightsT, errorMatrix)

			derivativeMatrix := matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.ApplyFunction(layer.Output, math.SigmoidDerivative, derivativeMatrix)

			layer.Deltas = matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.MultiplyElementWise(errorMatrix, derivativeMatrix, layer.Deltas)
		}
	}
}

func (n *Network) Update(learningRate float64) {
	for _, layer := range n.Layers {
		layer.Update(learningRate)
	}
}

func (n *Network) Train(inputs, expected *[]float64, learningRate float64) {
	n.Forward(inputs)
	n.Backward(expected)
	n.Update(learningRate)
}

func (n *Network) TrainLoop(input, expected [][]float64, learningRate float64, epoch int) {
	for i := 0; i < epoch; i++ {
		fmt.Printf("Epoch %d/%d\n", i+1, epoch)
		const barWidth = 50
		for j, in := range input {
			n.Train(&in, &expected[j], learningRate)

			if (j+1)%(len(input)/100) == 0 || j == len(input)-1 { // Update progress bar at ~1% intervals
				progress := float64(j+1) / float64(len(input))
				pos := int(progress * barWidth)
				bar := strings.Repeat("=", pos) + strings.Repeat(" ", barWidth-pos)
				fmt.Printf("\r[%%s] %.2f%%", bar, progress*100)
			}
		}
		fmt.Println() // Newline after progress bar is complete
	}
}

func (n *Network) getWeights() [][][]float64 {
	weights := make([][][]float64, len(n.Layers))
	for i, layer := range n.Layers {
		weights[i] = make([][]float64, layer.numNeurons)
		for j := 0; j < layer.numNeurons; j++ {
			neuronWeights := make([]float64, layer.numInputs)
			for k := 0; k < layer.numInputs; k++ {
				neuronWeights[k] = matrix.Get(k, j, layer.Weights)
			}
			weights[i][j] = neuronWeights
		}
	}
	return weights
}

func (n *Network) getBiases() [][]float64 {
	biases := make([][]float64, len(n.Layers))
	for i, layer := range n.Layers {
		biases[i] = layer.Biases.Data
	}
	return biases
}

func (n *Network) setWeights(weights [][][]float64) {
	for i, layer := range n.Layers {
		for j, neuronWeights := range weights[i] {
			for k, weight := range neuronWeights {
				matrix.Set(k, j, weight, layer.Weights)
			}
		}
	}
}

func (n *Network) setBiases(biases [][]float64) {
	for i, layer := range n.Layers {
		layer.Biases.Data = biases[i]
	}
}

func (n *Network) Save(path string) error {
	data := NetworkData{
		Weights: n.getWeights(),
		Biases:  n.getBiases(),
	}

	file, err := json.MarshalIndent(data, "", " ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, file, 0644)
}

func Load(path string) (*Network, error) {
	file, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var data NetworkData
	err = json.Unmarshal(file, &data)
	if err != nil {
		return nil, err
	}

	layerSizes := make([]int, len(data.Weights)+1)
	layerSizes[0] = len(data.Weights[0][0])
	for i, layerWeights := range data.Weights {
		layerSizes[i+1] = len(layerWeights)
	}

	n := NewNetwork(layerSizes)
	n.setWeights(data.Weights)
	n.setBiases(data.Biases)

	return n, nil
}

func (n *Network) Load(path string) error {
	file, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	var data NetworkData
	err = json.Unmarshal(file, &data)
	if err != nil {
		return err
	}

	n.setWeights(data.Weights)
	n.setBiases(data.Biases)

	return nil
}
