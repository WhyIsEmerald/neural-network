package geojson

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

// ExtractedGeoJSON holds the data extracted from the GeoJSON file
type ExtractedGeoJSON struct {
	MultiPolygons     []MultiPolygon
	MinLon, MinLat    float64
	MaxLon, MaxLat    float64
	FeatureCollection *FeatureCollection
}

// LoadAndExtractGeoJSON reads a GeoJSON file, extracts polygons, and calculates the bounding box.
func LoadAndExtractGeoJSON(path string) (*ExtractedGeoJSON, error) {
	file, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read geojson file: %w", err)
	}

	var featureCollection FeatureCollection
	if err := json.Unmarshal(file, &featureCollection); err != nil {
		return nil, fmt.Errorf("failed to unmarshal geojson: %w", err)
	}

	var multiPolygons []MultiPolygon
	minLon, minLat := 180.0, 90.0
	maxLon, maxLat := -180.0, -90.0

	for _, feature := range featureCollection.Features {
		var multiPolygon MultiPolygon
		switch feature.Geometry.Type {
		case "Polygon":
			var polygon Polygon
			if err := json.Unmarshal(feature.Geometry.Coordinates, &polygon); err != nil {
				return nil, fmt.Errorf("failed to unmarshal polygon: %w", err)
			}
			multiPolygon = MultiPolygon{polygon}
		case "MultiPolygon":
			if err := json.Unmarshal(feature.Geometry.Coordinates, &multiPolygon); err != nil {
				return nil, fmt.Errorf("failed to unmarshal multipolygon: %w", err)
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

	return &ExtractedGeoJSON{
		MultiPolygons:     multiPolygons,
		MinLon:            minLon,
		MinLat:            minLat,
		MaxLon:            maxLon,
		MaxLat:            maxLat,
		FeatureCollection: &featureCollection,
	}, nil
}


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
func IsPointInPolygon(point Point, polygon Polygon) bool {
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

func IsPointInMultiPolygon(point Point, multiPolygon MultiPolygon) bool {
	for _, polygon := range multiPolygon {
		if IsPointInPolygon(point, polygon) {
			return true
		}
	}
	return false
}
