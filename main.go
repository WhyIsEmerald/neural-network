package main

import (
	"fmt"

	"github.com/whyisemerald/neural_network/internals/network"
)

func main() {
	inputs := [][]float64{
		{0, 1},
		{1, 0},
		{1, 1},
		{0, 0},
	}
	expected := [][]float64{
		{1},
		{1},
		{0},
		{0},
	}
	n := network.NewNetwork([]int{2, 2, 1})
	n.TrainLoop(inputs, expected, 0.1, 1000000)
	fmt.Printf("Output for [0,1]: %v\n", n.Forward([]float64{0, 1})[0])
	fmt.Printf("Output for [1,0]: %v\n", n.Forward([]float64{1, 0})[0])
	fmt.Printf("Output for [1,1]: %v\n", n.Forward([]float64{1, 1})[0])
	fmt.Printf("Output for [0,0]: %v\n", n.Forward([]float64{0, 0})[0])
}
