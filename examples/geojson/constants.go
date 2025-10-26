package geojson

const (
	// GeojsonPath is the path to the GeoJSON file
	GeojsonPath = "data/IND_simple.geojson/IND_ADM1_simple.geojson"

	// ModelPath is the path to save the trained model
	ModelPath = "examples/geojson/saves/model.json"

	// NumSamples is the number of points to generate for training
	NumSamples = 10000

	// LearningRate is the learning rate for training
	LearningRate = 0.1

	// Epochs is the number of training cycles
	Epochs = 1000

	// TestCount is the number of test points to generate for accuracy testing
	TestCount = 1000
)

var (
	// HiddenLayerSizes is a slice of integers representing the size of each hidden layer
	HiddenLayerSizes = []int{50, 20}
)
