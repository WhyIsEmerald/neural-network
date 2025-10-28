package geojson

import (
	"fmt"
	"math/rand"
	"time"

	"slices"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Train() {
	rand.Seed(time.Now().UnixNano())

	// Load and extract GeoJSON data
	geoData, err := LoadAndExtractGeoJSON(GeojsonPath)
	if err != nil {
		panic(err)
	}

	// Generate training data
	inputs, expected := GenerateTrainingData(geoData, NumSamples)
	numClasses := len(geoData.FeatureCollection.Features)

	// Create or load the network
	var n *network.Network

	n, err = network.Load(ModelPath)

	layerSizes := []int{2}
	layerSizes = append(layerSizes, HiddenLayerSizes...)
	layerSizes = append(layerSizes, numClasses)

	if err == nil {
		fmt.Printf("Loaded existing model from %s. Continuing training.\n", ModelPath)
		if !slices.Equal(n.GetLayerSizes(), layerSizes) {
			fmt.Println("Model architecture has changed. Creating a new network.")
			n = network.NewNetwork(layerSizes)
		}
	} else {
		fmt.Println("No existing model found or failed to load. Creating a new network.")
		n = network.NewNetwork(layerSizes)
	}

	// Train the network
	fmt.Println("Training network...")
	n.TrainLoop(inputs, expected, LearningRate, Epochs)

	fmt.Println("Training complete.")

	// Save the model
	if err := n.Save(ModelPath); err != nil {
		panic(err)
	}
	fmt.Printf("Model saved to %s\n", ModelPath)
}
