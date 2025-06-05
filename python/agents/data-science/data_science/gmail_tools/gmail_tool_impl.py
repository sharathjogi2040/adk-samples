import logging
import re
from typing import List, Dict, Optional, Union # Added Union

from google.api_core.exceptions import GoogleAPIError
from google.cloud import discoveryengine_v1beta as discoveryengine
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _parse_datastore_path(datastore_path: str) -> Optional[Dict[str, str]]:
    """
    Parses a datastore path string into its components.
    Example path: projects/project_id/locations/location_id/collections/collection_id/dataStores/data_store_id
    """
    # Adjusted regex to be more flexible with global/regional locations and collection IDs
    match = re.match(
        r"projects/(?P<project_id>[^/]+)/locations/(?P<location>[^/]+)/"
        r"(collections/(?P<collection_id>[^/]+)/)?dataStores/(?P<data_store_id>[^/]+)",
        datastore_path,
    )
    if match:
        return match.groupdict()
    logger.warning(f"Datastore path format not recognized: {datastore_path}")
    return None

def query_vertex_ai_search_tool(
    datastore_path: str,
    search_query: str,
    max_results: int = 10
) -> List[Dict]:
    """
    Queries a Vertex AI Search datastore and retrieves email content.

    Args:
        datastore_path: The full resource path of the Vertex AI Search datastore.
                        e.g., projects/{project}/locations/{location}/collections/default_collection/dataStores/{data_store_id}
                        or projects/{project}/locations/{location}/dataStores/{data_store_id} (for some newer datastores)
        search_query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A list of dictionaries, where each dictionary represents an email document.
        Returns an empty list if an error occurs or no results are found.
    """
    logger.info(
        f"Executing query_vertex_ai_search_tool: datastore_path='{datastore_path}', "
        f"query='{search_query}', max_results={max_results}"
    )

    path_parts = _parse_datastore_path(datastore_path)
    if not path_parts:
        logger.error(f"Invalid datastore_path format or could not parse: {datastore_path}")
        return []

    project_id = path_parts["project_id"]
    location = path_parts["location"]
    data_store_id = path_parts["data_store_id"]
    # collection_id = path_parts.get("collection_id", "default_collection") # Default if not in path

    try:
        client = discoveryengine.SearchServiceClient()

        serving_config_path = client.serving_config_path(
            project=project_id,
            location=location,
            data_store=data_store_id,
            serving_config="default_search",
        )
        logger.info(f"Using serving_config_path: {serving_config_path}")

        request = discoveryengine.SearchRequest(
            serving_config=serving_config_path,
            query=search_query,
            page_size=max_results,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=1,
                    max_extractive_segment_count=1,
                    return_extractive_segment_score=True,
                )
            ),
        )

        response = client.search(request)

        processed_results: List[Dict] = []
        for result in response.results:
            doc = result.document
            doc_data = {}
            doc_data["id"] = doc.id
            doc_data["name"] = doc.name

            if doc.derived_struct_data and doc.derived_struct_data.fields:
                fields = doc.derived_struct_data.fields
                doc_data["subject"] = fields.get("subject", {}).string_value or fields.get("title", {}).string_value
                doc_data["sender"] = fields.get("sender", {}).string_value or fields.get("creator", {}).string_value or fields.get("author", {}).string_value
                doc_data["date"] = str(fields.get("created_time", {}) or fields.get("creationDate", {}) or fields.get("lastModified", {}))
                doc_data["full_content"] = fields.get("content", {}).string_value or fields.get("body", {}).string_value
            else:
                logger.warning(f"Document {doc.id} has no derived_struct_data or fields.")
                doc_data["full_content"] = "" # Ensure full_content exists

            doc_data["snippet"] = "" # Initialize snippet
            if result.extractive_answers:
                 doc_data["snippet"] = result.extractive_answers[0].content
            elif response.summary and response.summary.summary_with_metadata: # Fallback though current request doesn't prioritize summary
                 doc_data["snippet"] = response.summary.summary_with_metadata.summary

            if not doc_data.get("full_content") and doc_data.get("snippet"): # Use .get for safer access
                doc_data["full_content"] = doc_data["snippet"] # If no full_content, use snippet as a last resort

            processed_results.append(doc_data)

        logger.info(f"Found {len(processed_results)} results.")
        return processed_results

    except GoogleAPIError as e:
        logger.error(f"Google API Error during Vertex AI Search query for datastore {datastore_path}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during Vertex AI Search query for datastore {datastore_path}: {e}", exc_info=True)
        return []

# Example usage (for testing, not part of the agent tool itself)
# if __name__ == '__main__':
    # This is a placeholder and will not run in the agent's environment
    # logging.basicConfig(level=logging.INFO)
    # test_datastore_path = "projects/your-gcp-project/locations/global/collections/default_collection/dataStores/your-datastore-id"
    # test_query = "test query"
    # results = query_vertex_ai_search_tool(test_datastore_path, test_query)
    # for res in results:
    #    print(res)
    # pass

# --- Helper functions for email processing ---
def _strip_html_bs(html_content: str) -> str:
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.warning(f"BeautifulSoup failed to parse HTML: {e}", exc_info=False) # Keep log concise
        # Fallback: try to use regex to remove tags if BS fails or for simple cases
        text = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text) # Replace tags with space to avoid merging words
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text

def _remove_quoted_replies(text: str) -> str:
    if not text:
        return ""
    # Remove lines starting with ">" (common reply quote)
    text = re.sub(r"(?m)^>.*\n?", "", text)
    # Remove "On [Date/Time], [Name] <email> wrote:" lines
    text = re.sub(r"(?im)^on\s.+wrote:\n?", "", text, count=1) # Only the first occurrence usually
    # Remove lines like "From: ... Sent: ... To: ... Subject: ..." often found in forwarded messages
    text = re.sub(r"(?im)^from:.*\n(^to:.*\n)?(^cc:.*\n)?(^sent:.*\n)?(^subject:.*\n)?", "", text, count=1)
    return text

def _remove_signatures(text: str, signature_patterns: Optional[List[str]] = None) -> str:
    if not text:
        return ""

    common_patterns = [
        r"(?i)--\s*\n+[\s\S]*",  # Common "-- " signature separator
        r"(?i)(best regards|sincerely|regards|cheers|thanks|thank you)[,\s\n]+[\s\S]{0,150}", # Max 150 chars after greeting
        r"(?i)Sent from my iPhone",
        r"(?i)Sent from my Android",
        # Add more generic patterns if needed
    ]
    patterns_to_use = signature_patterns if signature_patterns else common_patterns

    # Split text into lines to better handle multiline signatures and avoid catastrophic backtracking
    lines = text.splitlines()
    cleaned_lines = []
    potential_signature_block = []
    signature_found_at = -1

    # Iterate backwards to find signature block
    for i in range(len(lines) - 1, -1, -1):
        line_to_check = "\n".join(lines[i:]) # Check from current line to end
        found_pattern = False
        for pattern in patterns_to_use:
            if re.search(pattern, line_to_check):
                # If a pattern matches from this line to the end, consider it a signature block
                signature_found_at = i
                found_pattern = True
                break
        if found_pattern:
            break

    if signature_found_at != -1:
        text = "\n".join(lines[:signature_found_at])

    return text

def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s*\n\s*", "\n", text)  # Normalize newlines (remove surrounding spaces)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)  # Replace multiple horizontal spaces with a single space
    text = text.strip()
    return text

# --- Main Tool Function ---
def process_email_content_tool(
    email_documents: Union[Dict, List[Dict]], # Corrected typing
    processing_level: str = "basic_clean",
    processing_options: Optional[Dict] = None
) -> Union[Dict, List[Dict]]: # Corrected typing
    """
    Processes raw email content to clean it and extract relevant text.

    Args:
        email_documents: A single email document dictionary or a list of such dictionaries.
                         Each dictionary must contain a 'full_content' field.
        processing_level: Currently supports "basic_clean".
        processing_options: Dictionary for options like:
                            {"remove_signatures": True/False (default True),
                             "signature_patterns": ["custom_regex_pattern", ...]}

    Returns:
        The processed email document(s) with a new 'cleaned_text' field and 'processing_log'.
    """
    if processing_options is None:
        processing_options = {}

    logger.info(f"Executing process_email_content_tool with level: {processing_level}")

    is_single_doc = isinstance(email_documents, dict)
    docs_to_process = [email_documents] if is_single_doc else email_documents

    processed_docs_list: List[Dict] = []

    for doc in docs_to_process:
        if not isinstance(doc, dict) or 'full_content' not in doc:
            logger.warning(f"Skipping invalid document: {doc}")
            processed_docs_list.append(doc) # Append as is if invalid
            continue

        current_text = str(doc.get('full_content', '')) # Ensure string
        processing_log: List[str] = []
        processed_doc = doc.copy() # Work on a copy

        if not current_text.strip():
            processing_log.append("Original content was empty or whitespace.")
            processed_doc['cleaned_text'] = ""
            processed_doc['processing_log'] = processing_log
            processed_docs_list.append(processed_doc)
            continue

        # 1. HTML stripping (always attempt, it's mostly safe)
        # Check if it looks like HTML before expensive parsing
        if '<' in current_text and '>' in current_text and ('<html' in current_text.lower() or '<body' in current_text.lower() or '<p>' in current_text.lower()):
            text_before_html_strip = current_text
            current_text = _strip_html_bs(current_text)
            if text_before_html_strip != current_text:
                 processing_log.append("Stripped HTML content.")

        if processing_level == "basic_clean":
            # 2. Remove quoted replies
            text_before_replies = current_text
            current_text = _remove_quoted_replies(current_text)
            if text_before_replies != current_text:
                processing_log.append("Attempted to remove quoted replies.")

            # 3. Remove signatures
            if processing_options.get("remove_signatures", True):
                text_before_signatures = current_text
                custom_patterns = processing_options.get("signature_patterns")
                current_text = _remove_signatures(current_text, signature_patterns=custom_patterns)
                if text_before_signatures != current_text:
                    processing_log.append("Attempted to remove signatures.")

            # 4. Normalize whitespace (as a final step)
            text_before_whitespace = current_text
            current_text = _normalize_whitespace(current_text)
            if text_before_whitespace != current_text or not processing_log: # Add log if it's the only change
                processing_log.append("Normalized whitespace.")

            processed_doc['cleaned_text'] = current_text

        elif processing_level == "extract_conversation":
            # Placeholder for more advanced logic. For now, performs basic_clean.
            # In future, this would involve more complex turn detection.
            logger.warning("Processing level 'extract_conversation' currently falls back to 'basic_clean'.")
            # (Perform basic_clean steps as above for fallback)
            current_text = _remove_quoted_replies(current_text)
            if processing_options.get("remove_signatures", True):
                 current_text = _remove_signatures(current_text, signature_patterns=processing_options.get("signature_patterns"))
            current_text = _normalize_whitespace(current_text)
            processed_doc['cleaned_text'] = current_text
            processing_log.append("Performed basic_clean as fallback for extract_conversation.")

        else:
            logger.warning(f"Unsupported processing_level: {processing_level}. Returning original content.")
            processed_doc['cleaned_text'] = current_text # Or original doc.get('full_content')
            processing_log.append(f"Unsupported processing_level: {processing_level}.")

        processed_doc['processing_log'] = processing_log
        processed_docs_list.append(processed_doc)

    return processed_docs_list[0] if is_single_doc else processed_docs_list

# Example usage (for testing)
if __name__ == '__main__':
    # This is a placeholder for local testing
    # logging.basicConfig(level=logging.INFO)
    # sample_html_email = {
    #     "full_content": "<html><body><p>Hello <b>World</b></p><div><!-- comment -->Text here.<br></div><style>p{color:red}</style>Regards,<br>John Doe<br>Sent from my iPhone</body></html>"
    # }
    # sample_text_email = {
    #     "full_content": "\n\nOn Tue, Jan 1, 2024 at 10:00 AM, Sender <sender@example.com> wrote:\n> This is a quoted reply.\n> It has multiple lines.\n\nThis is the actual new message.\n\n-- \nThanks,\nMy Name\nMy Title\n"
    # }
    # processed_html = process_email_content_tool(sample_html_email)
    # print("Processed HTML:", processed_html)
    # processed_text = process_email_content_tool(sample_text_email)
    # print("Processed Text:", processed_text)
    pass
