package main

import (
	"fmt"
	"time"

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
	start := time.Now()
	n := network.NewNetwork([]int{2, 2, 1})
	n.TrainLoop(inputs, expected, 0.1, 1000000)
	fmt.Printf("Training took: %v\n", time.Since(start))
	start = time.Now()
	input1 := []float64{0, 1}
	fmt.Printf("Output for [0,1]: %v\n", n.Forward(&input1)[0])
	input2 := []float64{1, 0}
	fmt.Printf("Output for [1,0]: %v\n", n.Forward(&input2)[0])
	input3 := []float64{1, 1}
	fmt.Printf("Output for [1,1]: %v\n", n.Forward(&input3)[0])
	input4 := []float64{0, 0}
	fmt.Printf("Output for [0,0]: %v\n", n.Forward(&input4)[0])
	fmt.Printf("Forward took: %v\n", time.Since(start))
}
