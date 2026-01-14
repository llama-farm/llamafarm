"""Tests for Keyword/Keyphrase Extraction.

Phase 7 of Universal Runtime ML Enhancements.
"""


from utils.keyword_extractor import STOPWORDS, KeywordExtractor


class TestKeywordExtractorBasics:
    """Test basic keyword extraction functionality."""

    def test_extract_returns_list(self):
        """Test that extraction returns a list."""
        extractor = KeywordExtractor()
        result = extractor.extract("Machine learning is great for data analysis.")
        assert isinstance(result, list)

    def test_extract_returns_keywords_with_scores(self):
        """Test that results have keyword and score."""
        extractor = KeywordExtractor()
        result = extractor.extract("Machine learning is great for data analysis.")
        assert len(result) > 0
        assert "keyword" in result[0]
        assert "score" in result[0]

    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty list."""
        extractor = KeywordExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_top_k_limits_results(self):
        """Test that top_k limits number of results."""
        extractor = KeywordExtractor()
        text = """
        Machine learning is a subset of artificial intelligence that focuses on
        developing algorithms and statistical models. Deep learning is a further
        subset of machine learning that uses neural networks with many layers.
        """
        result = extractor.extract(text, top_k=5)
        assert len(result) <= 5


class TestNgramGeneration:
    """Test n-gram candidate generation."""

    def test_unigrams(self):
        """Test extraction with unigrams only."""
        extractor = KeywordExtractor()
        result = extractor.extract("Python programming language", ngram_range=(1, 1))
        keywords = [r["keyword"] for r in result]
        # Should have single words
        for kw in keywords:
            assert len(kw.split()) == 1

    def test_bigrams(self):
        """Test extraction with bigrams."""
        extractor = KeywordExtractor()
        result = extractor.extract(
            "Machine learning and deep learning are popular",
            ngram_range=(2, 2)
        )
        keywords = [r["keyword"] for r in result]
        # Should have two-word phrases
        for kw in keywords:
            assert len(kw.split()) == 2

    def test_trigrams(self):
        """Test extraction with trigrams."""
        extractor = KeywordExtractor()
        result = extractor.extract(
            "Natural language processing is a field of artificial intelligence",
            ngram_range=(3, 3)
        )
        keywords = [r["keyword"] for r in result]
        for kw in keywords:
            assert len(kw.split()) == 3


class TestStopwordFiltering:
    """Test stopword handling."""

    def test_stopwords_filtered_at_edges(self):
        """Test that n-grams starting/ending with stopwords are filtered."""
        extractor = KeywordExtractor()
        result = extractor.extract("The machine and the learning")
        keywords = [r["keyword"] for r in result]

        # Should not start or end with stopwords
        for kw in keywords:
            words = kw.split()
            assert words[0] not in STOPWORDS
            assert words[-1] not in STOPWORDS

    def test_stopwords_constant_exists(self):
        """Test that STOPWORDS constant contains common words."""
        assert "the" in STOPWORDS
        assert "and" in STOPWORDS
        assert "is" in STOPWORDS
        assert "a" in STOPWORDS


class TestScoring:
    """Test keyword scoring."""

    def test_scores_between_zero_and_one(self):
        """Test that scores are normalized."""
        extractor = KeywordExtractor()
        result = extractor.extract("Machine learning deep learning neural networks")
        for r in result:
            assert 0 <= r["score"] <= 1.0

    def test_results_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        extractor = KeywordExtractor()
        result = extractor.extract(
            "Machine learning and artificial intelligence are transforming industries"
        )
        if len(result) > 1:
            scores = [r["score"] for r in result]
            assert scores == sorted(scores, reverse=True)


class TestDiversity:
    """Test diversity parameter."""

    def test_zero_diversity(self):
        """Test extraction with no diversity."""
        extractor = KeywordExtractor()
        result = extractor.extract(
            "Machine learning deep learning neural network learning",
            diversity=0.0
        )
        # Should work without errors
        assert len(result) > 0

    def test_max_diversity(self):
        """Test extraction with maximum diversity."""
        extractor = KeywordExtractor()
        result = extractor.extract(
            "Machine learning deep learning neural network learning",
            diversity=1.0
        )
        # Should work without errors
        assert len(result) > 0


class TestRealWorldTexts:
    """Test with realistic document texts."""

    def test_technical_document(self):
        """Test extraction from technical text."""
        text = """
        Convolutional neural networks (CNNs) are a class of deep learning models
        commonly used for image classification and object detection tasks. CNNs
        use convolutional layers to automatically learn spatial hierarchies of
        features from input images. The architecture typically includes pooling
        layers, fully connected layers, and activation functions like ReLU.
        """
        extractor = KeywordExtractor()
        result = extractor.extract(text, top_k=10)

        keywords = [r["keyword"] for r in result]
        # Should find relevant technical terms
        assert len(result) > 0
        # At least some multi-word phrases should be found
        has_multiword = any(len(kw.split()) > 1 for kw in keywords)
        assert has_multiword

    def test_news_article(self):
        """Test extraction from news-style text."""
        text = """
        The stock market experienced significant volatility today as investors
        reacted to the Federal Reserve's announcement about interest rate changes.
        Technology stocks led the decline, with major companies seeing their
        share prices drop by over 3 percent. Market analysts predict continued
        uncertainty in the coming weeks.
        """
        extractor = KeywordExtractor()
        result = extractor.extract(text, top_k=10)

        assert len(result) > 0
        keywords = [r["keyword"] for r in result]
        # Should find relevant finance/market terms
        assert any("stock" in kw or "market" in kw for kw in keywords)


class TestEdgeCases:
    """Test edge cases."""

    def test_very_short_text(self):
        """Test with very short text."""
        extractor = KeywordExtractor()
        result = extractor.extract("Hello")
        # May return empty or minimal results
        assert isinstance(result, list)

    def test_numbers_in_text(self):
        """Test text containing numbers."""
        extractor = KeywordExtractor()
        result = extractor.extract("The model achieved 95% accuracy on the test set")
        assert isinstance(result, list)

    def test_special_characters(self):
        """Test text with special characters."""
        extractor = KeywordExtractor()
        result = extractor.extract("Machine learning & AI are transforming tech!")
        assert isinstance(result, list)

    def test_unicode_text(self):
        """Test text with unicode characters."""
        extractor = KeywordExtractor()
        # Mixed English and accented characters
        result = extractor.extract("The café serves excellent café au lait")
        assert isinstance(result, list)

    def test_repeated_words(self):
        """Test text with repeated words."""
        extractor = KeywordExtractor()
        result = extractor.extract("data data data analysis analysis")
        assert len(result) > 0
