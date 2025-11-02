package network

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"time"

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
		// Use ReLU for hidden layers, Sigmoid for the output layer
		if i < len(layerSizes)-2 {
			layers[i] = NewLayer(layerSizes[i+1], layerSizes[i], math.Relu, math.ReluDerivative)
		} else {
			layers[i] = NewLayer(layerSizes[i+1], layerSizes[i], math.Sigmoid, math.SigmoidDerivative)
		}
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
			matrix.ApplyFunction(layer.Output, layer.activationDerivative, derivativeMatrix)

			layer.Deltas = matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.MultiplyElementWise(errorMatrix, derivativeMatrix, layer.Deltas)

		} else {
			nextLayer := n.Layers[i+1]

			weightsT := matrix.Transpose(nextLayer.Weights)
			errorMatrix := matrix.NewMatrix(nextLayer.Deltas.Rows, weightsT.Cols, make([]float64, nextLayer.Deltas.Rows*weightsT.Cols))
			matrix.DotProduct(nextLayer.Deltas, weightsT, errorMatrix)

			derivativeMatrix := matrix.NewMatrix(1, layer.numNeurons, make([]float64, layer.numNeurons))
			matrix.ApplyFunction(layer.Output, layer.activationDerivative, derivativeMatrix)

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
	const barWidth = 50
	startTime := time.Now()
	totalSamples := len(input) * epoch

	// Initial print of epoch bar and sample bar
	fmt.Printf("Epoch: [%s] %.2f%% (%d/%d) - Speed: -- samples/s - Time Left: --:--:--\n", strings.Repeat(" ", barWidth), 0.0, 0, epoch)
	fmt.Printf("Sample: [%s] %.2f%% (%d/%d)\n", strings.Repeat(" ", barWidth), 0.0, 0, len(input))

	for i := 0; i < epoch; i++ {
		for j, in := range input {
			n.Train(&in, &expected[j], learningRate)

			currentSample := i*len(input) + j + 1
			
			// Calculate time left and speed
			elapsedTime := time.Since(startTime)
			var timeLeft time.Duration = 0 // Initialize timeLeft
			var speed float64 = 0.0
			if currentSample > 0 {
				timePerSample := float64(elapsedTime) / float64(currentSample)
				remainingSamples := totalSamples - currentSample
				timeLeft = time.Duration(timePerSample * float64(remainingSamples))
				speed = float64(currentSample) / elapsedTime.Seconds()
			}

			// Update sample progress bar
			if (j+1)%(len(input)/100) == 0 || j == len(input)-1 { // Update sample bar at ~1% intervals
				// Move cursor up 2 lines, clear both lines, then redraw
				fmt.Print("\033[2A\033[K") 
				
				// Redraw epoch bar
		epochProgress := float64(i) / float64(epoch)
		epochBar := strings.Repeat("=", int(epochProgress*barWidth)) + strings.Repeat(" ", barWidth-int(epochProgress*barWidth))
			fmt.Printf("\rEpoch: [%s] %.2f%% (%d/%d) - Speed: %.2f samples/s - Time Left: %s\n", epochBar, epochProgress*100, i+1, epoch, speed, timeLeft.Round(time.Second))

				// Redraw sample bar
	sampleProgress := float64(j+1) / float64(len(input))
	sampleBar := strings.Repeat("=", int(sampleProgress*barWidth)) + strings.Repeat(" ", barWidth-int(sampleProgress*barWidth))
			fmt.Printf("\rSample: [%s] %.2f%% (%d/%d)\n", sampleBar, sampleProgress*100, j+1, len(input))
			}
		}
		// After each epoch, update epoch bar to reflect completion of current epoch
		// And ensure sample bar is 100% for the completed epoch
		fmt.Print("\033[2A\033[K") 
		epochProgress := float64(i+1) / float64(epoch)
		epochBar := strings.Repeat("=", int(epochProgress*barWidth)) + strings.Repeat(" ", barWidth-int(epochProgress*barWidth))
		
		// Recalculate timeLeft and speed for the end of the epoch
		elapsedTime := time.Since(startTime)
		var timeLeft time.Duration = 0
		var speed float64 = 0.0
		currentSample := (i+1)*len(input) // Total samples processed up to the end of this epoch
		if currentSample > 0 {
			timePerSample := float64(elapsedTime) / float64(currentSample)
			remainingSamples := totalSamples - currentSample
			timeLeft = time.Duration(timePerSample * float64(remainingSamples))
			speed = float64(currentSample) / elapsedTime.Seconds()
		}

		fmt.Printf("\rEpoch: [%s] %.2f%% (%d/%d) - Speed: %.2f samples/s - Time Left: %s\n", epochBar, epochProgress*100, i+1, epoch, speed, timeLeft.Round(time.Second))
		fmt.Printf("\rSample: [%s] 100.00%% (%d/%d)\n", strings.Repeat("=", barWidth), len(input), len(input))
	}
	// Final epoch bar (100%)
	fmt.Print("\033[2A\033[K") 
	fmt.Printf("\rEpoch: [%s] 100.00%% (%d/%d) - Speed: %.2f samples/s - Time Left: 0s\n", strings.Repeat("=", barWidth), epoch, epoch, float64(totalSamples)/time.Since(startTime).Seconds())
	fmt.Printf("\rSample: [%s] 100.00%% (%d/%d)\n", strings.Repeat("=", barWidth), len(input), len(input))
}

func (n *Network) GetLayerSizes() []int {
	layerSizes := make([]int, len(n.Layers)+1)
	if len(n.Layers) > 0 {
		layerSizes[0] = n.Layers[0].numInputs
		for i, layer := range n.Layers {
			layerSizes[i+1] = layer.numNeurons
		}
	}
	return layerSizes
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
