import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# Import GoogleVisionAPI from your google_vision module.
from google_vision import GoogleVisionAPI

logger = logging.getLogger(__name__)


class WebpageReaderConfiguration(BaseModel):
    """Configuration for the WebpageReader component."""
    request_timeout: int = 30
    max_content_length: int = 100000  # Limit content size to avoid processing huge pages
    wkhtmltoimage_path: str = "wkhtmltoimage"  # Command name or full path to wkhtmltoimage
    max_retries: int = 3
    retry_delay: float = 2.0


class WebpageReader:
    """A component that reads webpages using a screenshot and Google Vision OCR."""
    config_class = WebpageReaderConfiguration

    def __init__(self, config: Optional[WebpageReaderConfiguration] = None):
        self.config = config or WebpageReaderConfiguration()
        # Simple in-memory cache
        self._cache: Dict[str, str] = {}
        # Initialize Google Vision API (ensure GOOGLE_VISION_API_KEY is set)
        self.vision = GoogleVisionAPI()
        if not self.vision.check_available():
            logger.error("Google Vision API is not available. Please set the API key.")
        else:
            logger.info("Google Vision API initialized successfully.")

    def get_commands(self) -> Iterator[Dict]:
        # In your framework, you would yield Command objects.
        yield self.read_webpage

    def _capture_webpage_screenshot(self, url: str) -> Optional[str]:
        """
        Capture a screenshot of the webpage using wkhtmltoimage.
        Returns the path to the image file or None on failure.
        """
        for attempt in range(self.config.max_retries):
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp_file.close()
                cmd = [self.config.wkhtmltoimage_path, url, tmp_file.name]
                logger.info(f"Capturing screenshot with command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Optionally, check file size
                if os.path.getsize(tmp_file.name) > self.config.max_content_length:
                    logger.warning("Captured screenshot is too large; truncating may occur.")
                return tmp_file.name
            except Exception as e:
                logger.warning(f"wkhtmltoimage failed on attempt {attempt+1}/{self.config.max_retries}: {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        return None

    def _filter_content_for_topics(self, content: str, topics: List[str]) -> str:
        """Filter content to focus on specific topics."""
        lines = content.split("\n")
        title_line = lines[0] if lines else "# Webpage content"
        if not topics:
            return content
        topic_patterns = [re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE) for topic in topics]
        result = [title_line, "", f"Content filtered for topics: {', '.join(topics)}", ""]
        current_paragraph = []
        relevant_paragraphs = []
        for line in lines[1:]:
            line = line.strip()
            if not line and current_paragraph:
                paragraph = " ".join(current_paragraph)
                if any(pattern.search(paragraph) for pattern in topic_patterns):
                    relevant_paragraphs.append(paragraph)
                current_paragraph = []
            elif line:
                current_paragraph.append(line)
        if current_paragraph and any(pattern.search(" ".join(current_paragraph)) for pattern in topic_patterns):
            relevant_paragraphs.append(" ".join(current_paragraph))
        if not relevant_paragraphs:
            result.append("No content specifically matching the requested topics was found.")
        else:
            for i, paragraph in enumerate(relevant_paragraphs):
                result.append(f"Paragraph {i+1}:")
                result.append(paragraph)
                result.append("")
        return "\n".join(result)

    def read_webpage(self, url: str, topics_of_interest: Optional[List[str]] = None) -> str:
        """
        Read a webpage by capturing its screenshot and using Google Vision OCR to extract text.
        Optionally, filter the content for specified topics.
        
        Args:
            url: URL of the webpage to read.
            topics_of_interest: Optional list of topics to filter the content.
        
        Returns:
            The extracted text content or an error message.
        """
        logger.info(f"Reading webpage: {url}")

        # Check for cached content
        if url in self._cache:
            logger.info(f"Using cached content for {url}")
            content = self._cache[url]
            if topics_of_interest:
                return self._filter_content_for_topics(content, topics_of_interest)
            return content

        if not url.startswith(("http://", "https://")):
            return f"Invalid URL: {url}. URL must start with http:// or https://."

        # Capture a screenshot of the webpage
        screenshot_path = self._capture_webpage_screenshot(url)
        if not screenshot_path:
            return f"Failed to capture a screenshot of the webpage: {url}"

        logger.info(f"Screenshot saved to {screenshot_path}")

        # Use Google Vision's OCR (detect_text) to extract text from the screenshot
        vision_result = self.vision.detect_text(image_path=screenshot_path)
        if "error" in vision_result:
            os.remove(screenshot_path)
            return f"Error during OCR: {vision_result['error']}"

        extracted_text = vision_result.get("full_text", "").strip()
        if not extracted_text:
            # As a fallback, use the web detection feature to get a description
            web_result = self.vision.detect_web(image_path=screenshot_path)
            extracted_text = "\n".join(web_result.get("annotations", {}).get("best_guess_labels", []))
        
        # Clean up the temporary screenshot file
        try:
            os.remove(screenshot_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary screenshot: {e}")

        if not extracted_text:
            return "No text could be extracted from the webpage image."

        # Optionally filter for topics
        final_content = extracted_text
        if topics_of_interest:
            final_content = self._filter_content_for_topics(extracted_text, topics_of_interest)

        # Cache and return
        self._cache[url] = final_content
        return final_content
