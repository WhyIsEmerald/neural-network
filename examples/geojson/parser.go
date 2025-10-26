package geojson

import "encoding/json"

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
