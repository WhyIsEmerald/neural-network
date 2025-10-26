package geojson

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Train() {
	rand.Seed(time.Now().UnixNano())

	// Load GeoJSON data
	file, err := ioutil.ReadFile(GeojsonPath)
	if err != nil {
		panic(err)
	}

	var featureCollection FeatureCollection
	err = json.Unmarshal(file, &featureCollection)
	if err != nil {
		panic(err)
	}

	// Extract polygons and find bounding box
	var multiPolygons []MultiPolygon
	minLon, minLat := 180.0, 90.0
	maxLon, maxLat := -180.0, -90.0

	for _, feature := range featureCollection.Features {
		var multiPolygon MultiPolygon
		switch feature.Geometry.Type {
		case "Polygon":
			var polygon Polygon
			if err := json.Unmarshal(feature.Geometry.Coordinates, &polygon); err != nil {
				panic(err)
			}
			multiPolygon = MultiPolygon{polygon}
		case "MultiPolygon":
			if err := json.Unmarshal(feature.Geometry.Coordinates, &multiPolygon); err != nil {
				panic(err)
			}
		}

		for _, polygon := range multiPolygon {
			for _, ring := range polygon {
				for _, point := range ring {
					lon, lat := point[0], point[1]
					if lon < minLon {
						minLon = lon
					}
					if lon > maxLon {
						maxLon = lon
					}
					if lat < minLat {
						minLat = lat
					}
					if lat > maxLat {
						maxLat = lat
					}
				}
			}
		}
		multiPolygons = append(multiPolygons, multiPolygon)
	}

	// Generate training data
	inputs := make([][]float64, NumSamples)
	expected := make([][]float64, NumSamples)
	numClasses := len(featureCollection.Features)

	for i := 0; i < NumSamples; i++ {
		var point Point
		var regionIndex int
		for {
			lon := minLon + rand.Float64()*(maxLon-minLon)
			lat := minLat + rand.Float64()*(maxLat-minLat)
			point = Point{lon, lat}

			found := false
			for j, mp := range multiPolygons {
				if IsPointInMultiPolygon(point, mp) {
					regionIndex = j
					found = true
					break
				}
			}
			if found {
				break
			}
		}

		inputs[i] = []float64{point[0], point[1]}
		expected[i] = make([]float64, numClasses)
		expected[i][regionIndex] = 1
	}

	// Create and train the network
	fmt.Println("Training network...")
	layerSizes := []int{2}
	layerSizes = append(layerSizes, HiddenLayerSizes...)
	layerSizes = append(layerSizes, numClasses)
	n := network.NewNetwork(layerSizes)
	n.TrainLoop(inputs, expected, LearningRate, Epochs)

	fmt.Println("Training complete.")

	// Save the model
	if err := n.Save(ModelPath); err != nil {
		panic(err)
	}
	fmt.Printf("Model saved to %s", ModelPath)

}
