package geojson

import "math/rand"

func GenerateTrainingData(geoData *ExtractedGeoJSON, numSamples int) ([][]float64, [][]float64) {
	inputs := make([][]float64, numSamples)
	expected := make([][]float64, numSamples)
	numClasses := len(geoData.FeatureCollection.Features)

	for i := 0; i < numSamples; i++ {
		var point Point
		var regionIndex int
		for {
			lon := geoData.MinLon + rand.Float64()*(geoData.MaxLon-geoData.MinLon)
			lat := geoData.MinLat + rand.Float64()*(geoData.MaxLat-geoData.MinLat)
			point = Point{lon, lat}

			found := false
			for j, mp := range geoData.MultiPolygons {
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
	return inputs, expected
}
