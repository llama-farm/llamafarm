"""Date and time extraction using multiple parsing strategies."""

import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import logging

from rag.components.extractors.base import BaseExtractor
from rag.core.base import Document

logger = logging.getLogger(__name__)


class DateTimeExtractor(BaseExtractor):
    """
    Date and time extraction using multiple parsing strategies.
    
    Uses dateutil for sophisticated parsing with regex fallbacks.
    Extracts absolute dates, relative dates, and time expressions.
    """
    
    def __init__(self, name: str = "DateTimeExtractor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Configuration
        self.fuzzy_parsing = self.config.get("fuzzy_parsing", True)
        self.extract_relative = self.config.get("extract_relative", True)
        self.extract_times = self.config.get("extract_times", True)
        self.default_timezone = self.config.get("default_timezone", "UTC")
        self.date_formats = self.config.get("date_formats", [])
        
        # Try to import dateutil
        self.dateutil_available = self._check_dateutil()
        
        # Initialize regex patterns
        self.patterns = self._initialize_patterns()
        
        # Relative date mappings
        self.relative_mappings = self._initialize_relative_mappings()
    
    def _check_dateutil(self) -> bool:
        """Check if dateutil is available."""
        try:
            import dateutil.parser
            self.dateutil_parser = dateutil.parser
            return True
        except ImportError:
            self.logger.warning("dateutil not available, using regex-only parsing")
            return False
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for date/time extraction."""
        patterns = {}
        
        # Absolute date patterns
        patterns["date_slash"] = re.compile(
            r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b'
        )
        
        patterns["date_dash"] = re.compile(
            r'\b(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b'
        )
        
        patterns["date_written"] = re.compile(
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+'
            r'(?:19|20)?\d{2}\b',
            re.IGNORECASE
        )
        
        patterns["date_reverse_written"] = re.compile(
            r'\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+'
            r'(?:19|20)?\d{2}\b',
            re.IGNORECASE
        )
        
        # Time patterns
        patterns["time_12hr"] = re.compile(
            r'\b(?:0?[1-9]|1[0-2]):[0-5]\d(?::[0-5]\d)?\s*(?:AM|PM|am|pm)\b'
        )
        
        patterns["time_24hr"] = re.compile(
            r'\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b'
        )
        
        # Relative date patterns
        patterns["relative_days"] = re.compile(
            r'\b(?:yesterday|today|tomorrow|'
            r'(?:last|next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|'
            r'\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from\s+now|later))\b',
            re.IGNORECASE
        )
        
        patterns["relative_specific"] = re.compile(
            r'\b(?:in\s+)?(?:a\s+)?(?:few\s+)?(?:\d+\s+)?(?:days?|weeks?|months?|years?|hours?|minutes?)\s+'
            r'(?:ago|from\s+now|later|before|after)\b',
            re.IGNORECASE
        )
        
        # Combined datetime patterns
        patterns["datetime_combined"] = re.compile(
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'
            r'(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+'
            r'(?:19|20)?\d{2}\s+(?:at\s+)?'
            r'(?:0?[1-9]|1[0-2]):[0-5]\d(?::[0-5]\d)?\s*(?:AM|PM|am|pm)',
            re.IGNORECASE
        )
        
        return patterns
    
    def _initialize_relative_mappings(self) -> Dict[str, str]:
        """Initialize mappings for relative date expressions."""
        return {
            "yesterday": "-1 day",
            "today": "0 days",
            "tomorrow": "+1 day",
            "last week": "-1 week",
            "this week": "0 weeks",
            "next week": "+1 week",
            "last month": "-1 month",
            "this month": "0 months",
            "next month": "+1 month",
            "last year": "-1 year",
            "this year": "0 years",
            "next year": "+1 year",
        }
    
    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract dates and times from documents."""
        for doc in documents:
            try:
                datetime_data = self._extract_datetime_info(doc.content)
                
                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["datetime"] = datetime_data
                
                # Add simplified lists for easy access
                doc.metadata["dates"] = [dt["text"] for dt in datetime_data.get("dates", [])]
                doc.metadata["times"] = [dt["text"] for dt in datetime_data.get("times", [])]
                doc.metadata["relative_dates"] = [dt["text"] for dt in datetime_data.get("relative_dates", [])]
                
                total_extracted = (len(datetime_data.get("dates", [])) + 
                                 len(datetime_data.get("times", [])) + 
                                 len(datetime_data.get("relative_dates", [])))
                
                self.logger.debug(f"Extracted {total_extracted} datetime elements from document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"DateTime extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def _extract_datetime_info(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract comprehensive datetime information from text."""
        datetime_data = {
            "dates": [],
            "times": [],
            "relative_dates": [],
            "combined_datetime": []
        }
        
        # Extract absolute dates
        datetime_data["dates"].extend(self._extract_absolute_dates(text))
        
        # Extract times
        if self.extract_times:
            datetime_data["times"].extend(self._extract_times(text))
        
        # Extract relative dates
        if self.extract_relative:
            datetime_data["relative_dates"].extend(self._extract_relative_dates(text))
        
        # Extract combined datetime expressions
        datetime_data["combined_datetime"].extend(self._extract_combined_datetime(text))
        
        # Remove duplicates and overlaps
        datetime_data = self._deduplicate_extractions(datetime_data)
        
        return datetime_data
    
    def _extract_absolute_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract absolute date expressions."""
        dates = []
        
        # Use regex patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern_name.startswith("date_"):
                for match in pattern.finditer(text):
                    date_text = match.group().strip()
                    parsed_date = self._parse_date(date_text)
                    
                    if parsed_date:
                        dates.append({
                            "text": date_text,
                            "start": match.start(),
                            "end": match.end(),
                            "parsed": parsed_date.isoformat(),
                            "type": "absolute_date",
                            "pattern": pattern_name,
                            "confidence": 0.9
                        })
        
        # If dateutil is available, try additional parsing
        if self.dateutil_available and self.fuzzy_parsing:
            dates.extend(self._extract_fuzzy_dates(text))
        
        return dates
    
    def _extract_times(self, text: str) -> List[Dict[str, Any]]:
        """Extract time expressions."""
        times = []
        
        for pattern_name in ["time_12hr", "time_24hr"]:
            pattern = self.patterns[pattern_name]
            for match in pattern.finditer(text):
                time_text = match.group().strip()
                parsed_time = self._parse_time(time_text)
                
                if parsed_time:
                    times.append({
                        "text": time_text,
                        "start": match.start(),
                        "end": match.end(),
                        "parsed": parsed_time,
                        "type": "time",
                        "pattern": pattern_name,
                        "confidence": 0.9
                    })
        
        return times
    
    def _extract_relative_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract relative date expressions."""
        relative_dates = []
        
        for pattern_name in ["relative_days", "relative_specific"]:
            pattern = self.patterns[pattern_name]
            for match in pattern.finditer(text):
                relative_text = match.group().strip()
                interpretation = self._interpret_relative_date(relative_text)
                
                relative_dates.append({
                    "text": relative_text,
                    "start": match.start(),
                    "end": match.end(),
                    "interpretation": interpretation,
                    "type": "relative_date",
                    "pattern": pattern_name,
                    "confidence": 0.8
                })
        
        return relative_dates
    
    def _extract_combined_datetime(self, text: str) -> List[Dict[str, Any]]:
        """Extract combined date and time expressions."""
        combined = []
        
        pattern = self.patterns["datetime_combined"]
        for match in pattern.finditer(text):
            datetime_text = match.group().strip()
            parsed_datetime = self._parse_datetime(datetime_text)
            
            if parsed_datetime:
                combined.append({
                    "text": datetime_text,
                    "start": match.start(),
                    "end": match.end(),
                    "parsed": parsed_datetime.isoformat(),
                    "type": "combined_datetime",
                    "pattern": "datetime_combined",
                    "confidence": 0.95
                })
        
        return combined
    
    def _extract_fuzzy_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates using fuzzy parsing with dateutil."""
        fuzzy_dates = []
        
        # Split text into sentences to avoid false positives
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            try:
                # Use fuzzy parsing to find potential dates
                parsed_date, parse_info = self.dateutil_parser.parse(
                    sentence.strip(), 
                    fuzzy_with_tokens=True
                )
                
                # Extract the date portion that was actually parsed
                if parse_info:
                    remaining_tokens = parse_info[1]
                    original_length = len(sentence.strip().split())
                    parsed_length = len(remaining_tokens)
                    
                    # Only include if a significant portion was parsed as date
                    if parsed_length < original_length * 0.8:
                        fuzzy_dates.append({
                            "text": sentence.strip(),
                            "parsed": parsed_date.isoformat(),
                            "type": "fuzzy_date",
                            "pattern": "dateutil_fuzzy",
                            "confidence": 0.7,
                            "remaining_tokens": remaining_tokens
                        })
            
            except (ValueError, TypeError):
                # Fuzzy parsing failed, continue
                continue
        
        return fuzzy_dates
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse a date string into datetime object."""
        if self.dateutil_available:
            try:
                return self.dateutil_parser.parse(date_text)
            except (ValueError, TypeError):
                pass
        
        # Fallback to manual parsing for common formats
        formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d",
            "%m/%d/%y", "%m-%d-%y", "%y-%m-%d",
            "%B %d, %Y", "%b %d, %Y",
            "%d %B %Y", "%d %b %Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_text, fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_time(self, time_text: str) -> Optional[str]:
        """Parse a time string."""
        # Remove extra spaces
        time_text = ' '.join(time_text.split())
        
        # Convert to 24-hour format for consistency
        if 'am' in time_text.lower() or 'pm' in time_text.lower():
            try:
                if self.dateutil_available:
                    parsed = self.dateutil_parser.parse(time_text)
                    return parsed.strftime("%H:%M:%S")
                else:
                    # Manual AM/PM parsing
                    time_part = re.sub(r'\s*[ap]m\s*', '', time_text, flags=re.IGNORECASE)
                    is_pm = 'pm' in time_text.lower()
                    
                    if ':' in time_part:
                        parts = time_part.split(':')
                        hour = int(parts[0])
                        minute = int(parts[1])
                        second = int(parts[2]) if len(parts) > 2 else 0
                        
                        if is_pm and hour != 12:
                            hour += 12
                        elif not is_pm and hour == 12:
                            hour = 0
                        
                        return f"{hour:02d}:{minute:02d}:{second:02d}"
            except (ValueError, TypeError):
                pass
        
        return time_text  # Return as-is if parsing fails
    
    def _parse_datetime(self, datetime_text: str) -> Optional[datetime]:
        """Parse a combined datetime string."""
        if self.dateutil_available:
            try:
                return self.dateutil_parser.parse(datetime_text)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _interpret_relative_date(self, relative_text: str) -> str:
        """Interpret relative date expressions."""
        relative_lower = relative_text.lower().strip()
        
        # Check direct mappings
        for key, value in self.relative_mappings.items():
            if key in relative_lower:
                return value
        
        # Parse numeric relative expressions
        numbers = re.findall(r'\d+', relative_text)
        units = re.findall(r'(?:days?|weeks?|months?|years?|hours?|minutes?)', relative_text, re.IGNORECASE)
        directions = re.findall(r'(?:ago|from\s+now|later|before|after)', relative_text, re.IGNORECASE)
        
        if numbers and units:
            number = numbers[0]
            unit = units[0].lower().rstrip('s')  # Remove plural 's'
            
            direction = "+"
            if directions and any(d in ["ago", "before"] for d in directions):
                direction = "-"
            
            return f"{direction}{number} {unit}"
        
        return relative_text  # Return original if can't interpret
    
    def _deduplicate_extractions(self, datetime_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Remove duplicate and overlapping extractions."""
        all_extractions = []
        
        # Collect all extractions with positions
        for category, extractions in datetime_data.items():
            for extraction in extractions:
                if "start" in extraction and "end" in extraction:
                    extraction["category"] = category
                    all_extractions.append(extraction)
        
        # Sort by position
        all_extractions.sort(key=lambda x: x["start"])
        
        # Remove overlaps (keep highest confidence)
        filtered_extractions = []
        for extraction in all_extractions:
            is_overlap = False
            
            for existing in filtered_extractions:
                if (extraction["start"] < existing["end"] and 
                    extraction["end"] > existing["start"]):
                    # Overlapping - keep the one with higher confidence
                    if extraction["confidence"] > existing["confidence"]:
                        filtered_extractions.remove(existing)
                    else:
                        is_overlap = True
                    break
            
            if not is_overlap:
                filtered_extractions.append(extraction)
        
        # Reorganize by category
        result = {category: [] for category in datetime_data.keys()}
        for extraction in filtered_extractions:
            category = extraction.pop("category")
            result[category].append(extraction)
        
        return result
    
    def get_dependencies(self) -> List[str]:
        """Get optional dependencies."""
        return ["python-dateutil"]  # Optional dependency
    
    def validate_dependencies(self) -> bool:
        """Always return True as dateutil is optional."""
        return True  # Regex fallback always available