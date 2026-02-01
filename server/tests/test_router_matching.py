"""
Tests for router matching module.

Tests:
- Capability matching
- Route decision making
- Confidence thresholds
- Multi-capability routing
"""

import pytest

from router import CapabilityMatcher, Capability, MatchResult


@pytest.fixture
def sample_capabilities():
    """Create sample capabilities for testing."""
    return {
        "weather": Capability(
            name="weather",
            description="Get weather information and forecasts",
            examples=[
                "What's the weather in NYC?",
                "Will it rain tomorrow?",
                "Temperature forecast for this week"
            ],
            node_id="weather-001"
        ),
        "calculator": Capability(
            name="calculator",
            description="Perform mathematical calculations",
            examples=[
                "What is 15 times 23?",
                "Calculate the square root of 144",
                "Solve 5x + 3 = 18"
            ],
            node_id="math-001"
        ),
        "email": Capability(
            name="email",
            description="Send and manage emails",
            examples=[
                "Send an email to my boss",
                "Check my inbox",
                "Reply to the last message"
            ],
            node_id="email-001"
        )
    }


def test_capability_creation():
    """Test that capabilities can be created properly."""
    cap = Capability(
        name="test",
        description="Test capability",
        examples=["example 1", "example 2"],
        node_id="test-001"
    )
    
    assert cap.name == "test"
    assert cap.description == "Test capability"
    assert len(cap.examples) == 2
    assert cap.node_id == "test-001"


def test_matcher_initialization(sample_capabilities):
    """Test matcher initialization with capabilities."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    assert matcher is not None
    assert len(matcher.capabilities) == 3


def test_match_weather_query(sample_capabilities):
    """Test matching a weather query."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "What's the temperature going to be tomorrow?"
    results = matcher.match(query, top_k=1)
    
    assert len(results) > 0
    best_match = results[0]
    assert best_match.capability_name == "weather"
    assert best_match.confidence > 0.5


def test_match_calculator_query(sample_capabilities):
    """Test matching a calculator query."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "What is 42 divided by 7?"
    results = matcher.match(query, top_k=1)
    
    assert len(results) > 0
    best_match = results[0]
    assert best_match.capability_name == "calculator"


def test_match_email_query(sample_capabilities):
    """Test matching an email query."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "I need to send a message to the team"
    results = matcher.match(query, top_k=1)
    
    assert len(results) > 0
    best_match = results[0]
    assert best_match.capability_name == "email"


def test_top_k_matching(sample_capabilities):
    """Test that top_k returns correct number of results."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "What's the weather?"
    
    # Test different top_k values
    for k in [1, 2, 3]:
        results = matcher.match(query, top_k=k)
        assert len(results) <= k
        
        # Results should be sorted by confidence (descending)
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i+1].confidence


def test_confidence_threshold(sample_capabilities):
    """Test that confidence threshold filters results."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "What's the weather?"
    
    # High threshold should return fewer/no results
    results_high = matcher.match(query, threshold=0.9, top_k=3)
    results_low = matcher.match(query, threshold=0.1, top_k=3)
    
    # Low threshold should have more or equal results
    assert len(results_low) >= len(results_high)
    
    # All returned results should meet threshold
    for result in results_high:
        assert result.confidence >= 0.9


def test_match_result_structure():
    """Test that MatchResult has expected structure."""
    result = MatchResult(
        capability_name="test",
        confidence=0.85,
        node_id="test-001"
    )
    
    assert result.capability_name == "test"
    assert result.confidence == 0.85
    assert result.node_id == "test-001"


def test_no_match_below_threshold(sample_capabilities):
    """Test that very irrelevant queries return no matches with high threshold."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    # Query that doesn't match any capability well
    query = "xyz abc 123"
    results = matcher.match(query, threshold=0.9, top_k=3)
    
    # Should return empty or very low confidence
    for result in results:
        assert result.confidence < 0.9


def test_similar_capabilities_ranking():
    """Test ranking when multiple capabilities could match."""
    capabilities = {
        "weather_current": Capability(
            name="weather_current",
            description="Get current weather",
            examples=["What's the weather now?", "Current temperature"],
            node_id="weather-current-001"
        ),
        "weather_forecast": Capability(
            name="weather_forecast",
            description="Get weather forecast",
            examples=["What's the weather tomorrow?", "Weather this week"],
            node_id="weather-forecast-001"
        )
    }
    
    matcher = CapabilityMatcher(capabilities)
    
    # Query about current weather
    query = "What's the temperature right now?"
    results = matcher.match(query, top_k=2)
    
    assert len(results) == 2
    # Current weather should rank higher
    assert results[0].capability_name == "weather_current"


def test_empty_query(sample_capabilities):
    """Test handling of empty query."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    results = matcher.match("", top_k=3)
    
    # Should still return results, just with low confidence
    assert isinstance(results, list)


def test_special_characters_in_query(sample_capabilities):
    """Test handling of special characters."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    query = "What's 2+2? 🤔"
    results = matcher.match(query, top_k=1)
    
    assert len(results) > 0
    # Should match calculator
    assert results[0].capability_name == "calculator"


def test_capability_with_no_examples():
    """Test capability with minimal examples."""
    cap = Capability(
        name="minimal",
        description="A minimal capability",
        examples=["one example"],
        node_id="min-001"
    )
    
    capabilities = {"minimal": cap}
    matcher = CapabilityMatcher(capabilities)
    
    # Should still work
    results = matcher.match("one example", top_k=1)
    assert len(results) > 0


def test_multiple_queries_same_capability(sample_capabilities):
    """Test that similar queries match to same capability."""
    matcher = CapabilityMatcher(sample_capabilities)
    
    weather_queries = [
        "What's the weather?",
        "Is it raining?",
        "Temperature today?",
        "Weather forecast"
    ]
    
    for query in weather_queries:
        results = matcher.match(query, top_k=1)
        assert len(results) > 0
        # All should match weather capability
        assert results[0].capability_name == "weather"
