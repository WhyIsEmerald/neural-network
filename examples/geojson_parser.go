package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

// GeoJSON structs
type Point []float64

type Polygon [][][]float64

type MultiPolygon [][][][]float64

type Geometry struct {
	Type        string          `json:"type"`
	Coordinates json.RawMessage `json:"coordinates"`
}

type Feature struct {
	Type       string                 `json:"type"`
	Geometry   Geometry               `json:"geometry"`
	Properties map[string]interface{} `json:"properties"`
}

type FeatureCollection struct {
	Type     string    `json:"type"`
	Features []Feature `json:"features"`
}

// Point in polygon check (Ray Casting algorithm)
func isPointInPolygon(point Point, polygon Polygon) bool {
	in := false
	for _, ring := range polygon {
		for i, j := 0, len(ring)-1; i < len(ring); j, i = i, i+1 {
			p1 := ring[i]
			p2 := ring[j]
			if (p1[1] > point[1]) != (p2[1] > point[1]) &&
				(point[0] < (p2[0]-p1[0])*(point[1]-p1[1])/(p2[1]-p1[1])+p1[0]) {
				in = !in
			}
		}
	}
	return in
}

func isPointInMultiPolygon(point Point, multiPolygon MultiPolygon) bool {
	for _, polygon := range multiPolygon {
		if isPointInPolygon(point, polygon) {
			return true
		}
	}
	return false
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Load GeoJSON data
	filePath := "data/IND_simple.geojson/IND_ADM1_simple.geojson"
	file, err := ioutil.ReadFile(filePath)
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
	numSamples := 1000
	inputs := make([][]float64, numSamples)
	expected := make([][]float64, numSamples)
	numClasses := len(featureCollection.Features)

	for i := 0; i < numSamples; i++ {
		var point Point
		var regionIndex int
		for {
			lon := minLon + rand.Float64()*(maxLon-minLon)
			lat := minLat + rand.Float64()*(maxLat-minLat)
			point = Point{lon, lat}

			found := false
			for j, mp := range multiPolygons {
				if isPointInMultiPolygon(point, mp) {
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
	n := network.NewNetwork([]int{2, 16, numClasses})
	n.TrainLoop(inputs, expected, 0.1, 100)

	fmt.Println("Training complete.")

	// Test with a random point
	lon := minLon + rand.Float64()*(maxLon-minLon)
	lat := minLat + rand.Float64()*(maxLat-minLat)
	testPoint := []float64{lon, lat}
	output := n.Forward(&testPoint)

	maxVal := -1.0
	predictedRegion := -1
	for i, val := range output {
		if val > maxVal {
			maxVal = val
			predictedRegion = i
		}
	}

	fmt.Printf("Test point: [%f, %f]\n", lon, lat)
	if predictedRegion != -1 {
		fmt.Printf("Predicted region: %s\n", featureCollection.Features[predictedRegion].Properties["NAME"])
	} else {
		fmt.Println("Could not predict region.")
	}
}
