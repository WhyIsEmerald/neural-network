package geojson

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Test() {
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
			lon := minLon + rand.Float64()*(maxLon-minLon)
			lat := minLat + rand.Float64()*(maxLat-minLat)
			point = Point{lon, lat}

			found := false
			for j, mp := range multiPolygons {
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
