package cmd

import "testing"

func TestIsLocalhost(t *testing.T) {
    cases := []struct{
        in  string
        yes bool
    }{
        {"http://localhost:8000", true},
        {"http://127.0.0.1:8000", true},
        {"http://::1:8000", true},
        {"http://example.com", false},
    }
    for _, c := range cases {
        if isLocalhost(c.in) != c.yes {
            t.Fatalf("isLocalhost(%q) mismatch; expected %v", c.in, c.yes)
        }
    }
}
