package geojson

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/whyisemerald/neural_network/internals/network"
)

func Forward() {
	rand.Seed(time.Now().UnixNano())

	// Load the model
	n, err := network.Load(ModelPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded model from %s\n", ModelPath)

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("Enter longitude (e.g., -122.4194) or 'q' to quit: ")
		lonStr, _ := reader.ReadString('\n')
		lonStr = strings.TrimSpace(lonStr)
		if lonStr == "q" {
			break
		}
		lon, err := strconv.ParseFloat(lonStr, 64)
		if err != nil {
			fmt.Println("Invalid longitude. Please enter a number.")
			continue
		}

		fmt.Print("Enter latitude (e.g., 37.7749): ")
		latStr, _ := reader.ReadString('\n')
		latStr = strings.TrimSpace(latStr)
		lat, err := strconv.ParseFloat(latStr, 64)
		if err != nil {
			fmt.Println("Invalid latitude. Please enter a number.")
			continue
		}

		input := []float64{lon, lat}

		// Perform a forward pass
		output := n.Forward(&input)

		// Find the region with the highest probability
		maxProb := -1.0
		predictedRegion := -1
		for i, prob := range output {
			if prob > maxProb {
				maxProb = prob
				predictedRegion = i
			}
		}

		fmt.Printf("Input coordinates: (%.4f, %.4f)\n", lon, lat)
		fmt.Printf("Predicted region index: %d\n", predictedRegion)
		fmt.Printf("Probability: %.2f%%\n", maxProb*100)
		fmt.Println("------------------------------------")
	}
}