import asyncio
import logging
import os
import re
import time
import aiohttp
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
from urllib.parse import urljoin

from pydantic import BaseModel
from bs4 import BeautifulSoup

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.url_validator import validate_url

from .google_cse import google_cse_search
from components.google_vision import GoogleVisionAPI

logger = logging.getLogger(__name__)


class WebCSEConfiguration(BaseModel):
    max_results: int = 3
    max_retries: int = 3
    retry_delay: float = 2.0
    max_images_to_process: int = 5  # Limit the number of images to process for text detection


class WebCSEComponent(DirectiveProvider, CommandProvider, ConfigurableComponent[WebCSEConfiguration]):
    """
    A component that reads webpage content by using Google Custom Search Engine
    to extract information from webpages.
    """

    config_class = WebCSEConfiguration

    def __init__(self, llm_provider, data_dir: Path, config: Optional[WebCSEConfiguration] = None):
        # This constructor accepts llm_provider and data_dir for compatibility with your agent.
        ConfigurableComponent.__init__(self, config)
        self.llm_provider = llm_provider
        self.data_dir = data_dir
        self.config = config or WebCSEConfiguration()
        self._cache = {}  # In-memory cache for URL contents
        self.session = None  # Will be initialized when needed

    async def _ensure_session(self):
        """Ensure that an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _close_session(self):
        """Close the aiohttp session if it exists."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def get_resources(self) -> Iterator[str]:
        yield "Ability to read webpages via Google Custom Search Engine."
        yield "Ability to detect text in images on webpages using Google Vision API."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_webpage

    @command(
        ["read_webpage"],
        "Read a webpage by extracting its content using Google Custom Search Engine.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL of the webpage to read or search query related to the content.",
                required=True,
            ),
            "topics_of_interest": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="Optional list of topics to filter the extracted text.",
                required=False,
            ),
            "detect_text_from_images": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether to use Google Vision API to detect text from images on the webpage.",
                required=False,
            ),
            "language_hints": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="Optional list of language hints for text detection (e.g., 'en', 'fr').",
                required=False,
            ),
        },
    )
    @validate_url
    async def read_webpage(
        self, 
        url: str, 
        topics_of_interest: Optional[List[str]] = None,
        detect_text_from_images: bool = False,
        language_hints: Optional[List[str]] = None
    ) -> str:
        """
        Read a webpage by using Google Custom Search Engine to extract content.
        Optionally, filter the text for specific topics or detect text from images.
        
        Args:
            url: URL of the webpage to read or a search query.
            topics_of_interest: Optional list of topics to filter the extracted text.
            detect_text_from_images: Whether to use Google Vision API to detect text from images.
            language_hints: Optional list of language hints for text detection.
        
        Returns:
            The extracted text or an error message.
        """
        try:
            logger.info(f"Reading webpage content for: {url}")

            # First try to get content using Google CSE
            extracted_text = ""
            for attempt in range(self.config.max_retries):
                try:
                    # Use the url as a search query to get information about it
                    search_query = f"site:{url}" if "://" in url else url
                    extracted_text = google_cse_search(search_query, self.config.max_results)
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{self.config.max_retries} failed: {e}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    extracted_text = f"Failed to retrieve content for: {url}"

            if not extracted_text or "Error" in extracted_text:
                return f"Failed to retrieve content for: {url}"

            # Apply topic filtering if specified
            if topics_of_interest:
                extracted_text = self._filter_content_for_topics(extracted_text, topics_of_interest)

            # Add text detection from images if requested
            if detect_text_from_images:
                try:
                    # Only attempt to extract images if the URL is a proper web URL
                    if "://" in url:
                        logger.info(f"Attempting to detect text from images on: {url}")
                        
                        # Extract image URLs
                        image_urls = await self._extract_image_urls(url)
                        
                        if image_urls:
                            # Limit the number of images to process
                            image_urls = image_urls[:self.config.max_images_to_process]
                            
                            # Initialize Google Vision API
                            vision_api = GoogleVisionAPI(self.data_dir)
                            
                            if not vision_api.check_available():
                                extracted_text += "\n\nNote: Google Vision API is not available. Unable to detect text from images."
                            else:
                                image_text_section = "\n\n## Text Detected from Images:\n"
                                has_detected_text = False
                                
                                # Process each image
                                for img_url in image_urls:
                                    try:
                                        # Download image
                                        img_data = await self._download_image(img_url)
                                        if not img_data:
                                            continue
                                        
                                        # Detect text
                                        text_result = vision_api.detect_text(
                                            image_bytes=img_data,
                                            language_hints=language_hints
                                        )
                                        
                                        if text_result.get("success", False) and "full_text" in text_result:
                                            full_text = text_result["full_text"].strip()
                                            if full_text:
                                                image_text_section += f"\n### Image from {img_url}:\n{full_text}\n"
                                                has_detected_text = True
                                    except Exception as e:
                                        logger.warning(f"Failed to process image {img_url}: {e}")
                                
                                # Add detected text to the extracted content
                                if has_detected_text:
                                    extracted_text += image_text_section
                                else:
                                    extracted_text += "\n\nNote: No text was detected in the images on this webpage."
                        else:
                            extracted_text += "\n\nNote: No images were found on this webpage for text detection."
                    else:
                        extracted_text += "\n\nNote: Text detection from images is only available for direct web URLs."
                except Exception as e:
                    logger.error(f"Error in text detection from images: {e}")
                    extracted_text += "\n\nNote: An error occurred while attempting to detect text from images."
                finally:
                    # Ensure the session is closed
                    await self._close_session()

            return extracted_text
        except Exception as e:
            logger.error(f"Error in read_webpage: {e}")
            await self._close_session()
            return f"Error reading webpage: {str(e)}"

    def _filter_content_for_topics(self, content: str, topics: List[str]) -> str:
        """
        Filter the extracted content to only include paragraphs that mention any of the specified topics.
        """
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

    async def _extract_image_urls(self, url: str) -> List[str]:
        """
        Extract image URLs from a webpage.
        
        Args:
            url: The URL of the webpage.
            
        Returns:
            A list of image URLs.
        """
        try:
            # Ensure session is available
            session = await self._ensure_session()
            
            # Fetch webpage content
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch webpage: {url}, status: {response.status}")
                    return []
                
                html_content = await response.text()
            
            # Parse HTML and extract image URLs
            soup = BeautifulSoup(html_content, 'html.parser')
            image_urls = []
            
            # Find all img tags
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    # Handle relative URLs
                    if not src.startswith(('http://', 'https://')):
                        src = urljoin(url, src)
                    
                    # Filter out small icons, data URIs, etc.
                    if (src.startswith(('http://', 'https://')) and 
                        not src.endswith(('.ico', '.svg')) and
                        not src.startswith('data:')):
                        image_urls.append(src)
            
            logger.info(f"Found {len(image_urls)} images on {url}")
            return image_urls
        except Exception as e:
            logger.error(f"Error extracting image URLs from {url}: {e}")
            return []

    async def _download_image(self, url: str) -> Optional[bytes]:
        """
        Download an image from a URL.
        
        Args:
            url: The URL of the image.
            
        Returns:
            The image data as bytes or None if download failed.
        """
        try:
            # Ensure session is available
            session = await self._ensure_session()
            
            # Download image
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download image: {url}, status: {response.status}")
                    return None
                
                # Read image data
                image_data = await response.read()
                
                # Check if it's actually an image (simple check)
                if len(image_data) < 100:  # Too small to be a valid image
                    logger.warning(f"Downloaded file is too small to be an image: {url}")
                    return None
                
                logger.info(f"Successfully downloaded image: {url}, size: {len(image_data)} bytes")
                return image_data
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None