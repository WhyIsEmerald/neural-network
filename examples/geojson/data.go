package geojson

import "math/rand"

// GenerateTrainingData creates training samples from extracted GeoJSON data.
func GenerateTrainingData(geoData *ExtractedGeoJSON, numSamples int) ([][]float64, [][]float64) {
	inputs := make([][]float64, numSamples)
	expected := make([][]float64, numSamples)
	numClasses := len(geoData.FeatureCollection.Features)

	for i := 0; i < numSamples; i++ {
		var point Point
		var regionIndex int
		for {
			// Generate a random point within the bounding box
			lon := geoData.MinLon + rand.Float64()*(geoData.MaxLon-geoData.MinLon)
			lat := geoData.MinLat + rand.Float64()*(geoData.MaxLat-geoData.MinLat)
			point = Point{lon, lat}

			// Check which polygon the point falls into
			found := false
			for j, mp := range geoData.MultiPolygons {
				if IsPointInMultiPolygon(point, mp) {
					regionIndex = j
					found = true
					break
				}
			}
			if found {
				break // Found a point inside a polygon, exit the loop
			}
		}

		// Create the input vector (the coordinates)
		inputs[i] = []float64{point[0], point[1]}
		// Create the expected output vector (one-hot encoded)
		expected[i] = make([]float64, numClasses)
		expected[i][regionIndex] = 1
	}
	return inputs, expected
}
