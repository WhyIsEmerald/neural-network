package geojson

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Forward() {
	rand.Seed(time.Now().UnixNano())

	// Load the model
	n, err := network.Load(ModelPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded model from %s\n", ModelPath)

	// Create a random point for prediction
	lon := -122.4194 + rand.Float64()*2 - 1 // Example longitude around San Francisco
	lat := 37.7749 + rand.Float64()*2 - 1   // Example latitude around San Francisco
	input := []float64{lon, lat}

	// Perform a forward pass
	output := n.Forward(&input)

	// Find the region with the highest probability
	maxProb := -1.0
	predictedRegion := -1
	for i, prob := range output {
		if prob > maxProb {
			maxProb = prob
			predictedRegion = i
		}
	}

	fmt.Printf("Input coordinates: (%.4f, %.4f)\n", lon, lat)
	fmt.Printf("Predicted region index: %d\n", predictedRegion)
	fmt.Printf("Probability: %.2f%%\n", maxProb*100)
}

