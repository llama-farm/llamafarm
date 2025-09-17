package cmd

import "testing"

func TestBuildServerURL(t *testing.T) {
    tests := []struct{
        name string
        base string
        path string
        want string
    }{
        {"no_trailing_base_with_leading_path", "http://localhost:8000", "/v1/info", "http://localhost:8000/v1/info"},
        {"trailing_base_with_leading_path", "http://localhost:8000/", "/v1/info", "http://localhost:8000/v1/info"},
        {"trailing_base_without_leading_path", "http://localhost:8000/", "v1/info", "http://localhost:8000/v1/info"},
        {"no_path", "http://localhost:8000/", "", "http://localhost:8000"},
    }
    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            got := buildServerURL(tc.base, tc.path)
            if got != tc.want {
                t.Fatalf("buildServerURL(%q,%q) = %q; want %q", tc.base, tc.path, got, tc.want)
            }
        })
    }
}
