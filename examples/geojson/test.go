package geojson

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Test() {
	rand.Seed(time.Now().UnixNano())

	// Load and extract GeoJSON data
	geoData, err := LoadAndExtractGeoJSON(GeojsonPath)
	if err != nil {
		panic(err)
	}

	// Load the trained model
	n, err := network.Load(ModelPath)
	if err != nil {
		panic(err)
	}

	// Test the network
	fmt.Println("Testing network...")
	correctPredictions := 0
	for i := 0; i < TestCount; i++ {
		var point Point
		var actualRegionIndex int
		for {
			lon := geoData.MinLon + rand.Float64()*(geoData.MaxLon-geoData.MinLon)
			lat := geoData.MinLat + rand.Float64()*(geoData.MaxLat-geoData.MinLat)
			point = Point{lon, lat}

			found := false
			for j, mp := range geoData.MultiPolygons {
				if IsPointInMultiPolygon(point, mp) {
					actualRegionIndex = j
					found = true
					break
				}
			}
			if found {
				break
			}
		}

		output := n.Forward(&[]float64{point[0], point[1]})
		predictedRegionIndex := 0
		maxOutput := -1.0
		for j, val := range output {
			if val > maxOutput {
				maxOutput = val
				predictedRegionIndex = j
			}
		}

		if actualRegionIndex == predictedRegionIndex {
			correctPredictions++
		}
	}

	accuracy := float64(correctPredictions) / float64(TestCount) * 100
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)
}
