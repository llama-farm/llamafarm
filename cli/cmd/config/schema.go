package config

import (
	"embed"
	"fmt"
	"log"

	"gopkg.in/yaml.v2"
)

//go:generate echo "ATTENTION: Running file generator..." && go run ./tools/copy ../config/schema.yaml cmd/config/schema.yaml

//go:embed schema.yaml
var configSchema embed.FS

// Schema represents a basic JSON/YAML schema structure.
// It's simplified to focus on the generation logic.
type Schema struct {
	Type        string             `yaml:"type"`
	Properties  map[string]*Schema `yaml:"properties,omitempty"`
	Required    []string           `yaml:"required,omitempty"`
	Items       *Schema            `yaml:"items,omitempty"`
	Default     any                `yaml:"default,omitempty"`
	Example     any                `yaml:"example,omitempty"`
	Description string             `yaml:"description,omitempty"`
}

// generateFromSchema recursively generates a Go map from a given schema.
// This map can then be marshaled into a YAML file.
func generateFromSchema(schema *Schema) (interface{}, error) {
	// Priority 1: Use the default value if provided.
	if schema.Default != nil {
		return schema.Default, nil
	}

	// Priority 2: Use the example value if provided.
	if schema.Example != nil {
		return schema.Example, nil
	}

	// Priority 3: Generate based on the type.
	switch schema.Type {
	case "object":
		// For objects, iterate through required properties and generate a value for each.
		obj := make(map[string]any)
		if schema.Properties == nil {
			return obj, nil // Return empty object if no properties are defined
		}
		for _, key := range schema.Required {
			val, err := generateFromSchema(schema.Properties[key])
			if err != nil {
				return nil, fmt.Errorf("error generating property '%s': %w", key, err)
			}
			obj[key] = val
		}
		return obj, nil

	case "array":
		// For arrays, generate one sample item based on the 'items' schema.
		if schema.Items == nil {
			return []any{}, nil // Return empty array if 'items' is not defined
		}
		item, err := generateFromSchema(schema.Items)
		if err != nil {
			return nil, fmt.Errorf("error generating array item: %w", err)
		}
		return []any{item}, nil

	case "string":
		// Provide a placeholder string.
		return "example_string", nil

	case "integer", "number":
		// Provide a placeholder number.
		return 123, nil

	case "boolean":
		// Provide a placeholder boolean.
		return true, nil

	default:
		return nil, fmt.Errorf("unsupported schema type: '%s'", schema.Type)
	}
}

func Generate() (string, error) {
	schemaData, err := configSchema.ReadFile("schema.yaml")
	if err != nil {
		log.Fatalf("Failed to read schema file '%s': %v", "schema.yaml", err)
	}

	// Unmarshal the YAML schema into our Go struct.
	var schema Schema
	err = yaml.Unmarshal(schemaData, &schema)
	if err != nil {
		log.Fatalf("Failed to unmarshal schema YAML: %v", err)
	}

	// Generate the data structure from the schema.
	generatedData, err := generateFromSchema(&schema)
	if err != nil {
		log.Fatalf("Failed to generate data from schema: %v", err)
	}

	// Marshal the generated data into a YAML string.
	outputYAML, err := yaml.Marshal(generatedData)
	if err != nil {
		log.Fatalf("Failed to marshal generated data to YAML: %v", err)
	}

	return string(outputYAML), nil
}
