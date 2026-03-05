package cmd

import "embed"

//go:embed all:registry
var embeddedRegistry embed.FS
