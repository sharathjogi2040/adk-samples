import logging
import re
from typing import List, Dict, Optional, Union
from datetime import datetime # Add for date parsing

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
            doc_data["name"] = doc.name # Resource name of the document

            if doc.derived_struct_data and doc.derived_struct_data.fields:
                fields = doc.derived_struct_data.fields

                # Title / Subject / Name / Summary
                doc_data["title"] = (
                    fields.get("subject", {}).string_value or
                    fields.get("title", {}).string_value or
                    fields.get("name", {}).string_value or # Often used for file names
                    fields.get("summary", {}).string_value or # For Calendar events
                    ""
                )

                # Sender / Author / Creator / Organizer
                organizer_field = fields.get("organizer", {})
                doc_data["author"] = (
                    fields.get("sender", {}).string_value or
                    fields.get("author", {}).string_value or
                    fields.get("creator", {}).string_value or
                    organizer_field.string_value or # If organizer is a simple string
                    (organizer_field.struct_value.fields.get("email",{}).string_value if organizer_field.struct_value and organizer_field.struct_value.fields else "") or
                    ""
                )

                # Date (prefer more specific, fallback to general)
                # Convert potential timestamp objects or date strings to string
                date_val = (
                    fields.get("created_time", {}) or
                    fields.get("creationDate", {}) or # Common in some schemas
                    fields.get("lastModified", {}) or
                    fields.get("modifiedTime", {}) or # Common in Drive
                    fields.get("startTime", {}) or # For Calendar events
                    fields.get("date", {})
                )

                if date_val.string_value:
                    doc_data["date"] = date_val.string_value
                elif date_val.number_value: # Handle epoch/timestamps if they appear as numbers
                    doc_data["date"] = str(date_val.number_value)
                elif date_val.struct_value and date_val.struct_value.fields.get("value",{}).number_value: # e.g. for Calendar start/end
                     doc_data["date"] = str(date_val.struct_value.fields.get("value",{}).number_value) # Assuming timestamp
                elif date_val.struct_value and date_val.struct_value.fields.get("date",{}).string_value: # e.g. for Calendar all-day event start/end date
                     doc_data["date"] = str(date_val.struct_value.fields.get("date",{}).string_value)
                else:
                    # Attempt to stringify the raw Value object if no specific type matched
                    # This is a fallback and might produce "[type_name]: [value]"
                    doc_data["date"] = str(date_val) if date_val and (date_val.string_value or date_val.number_value or date_val.struct_value) else ""


                # Main Content / Body / Description
                doc_data["full_content"] = (
                    fields.get("content", {}).string_value or
                    fields.get("body", {}).string_value or
                    fields.get("description", {}).string_value or # Common for Calendar events, Drive file descriptions
                    ""
                )

                # Link / URI
                doc_data["link"] = (
                    fields.get("webViewLink", {}).string_value or # Drive
                    fields.get("htmlLink", {}).string_value or    # Calendar
                    fields.get("uri", {}).string_value or
                    ""
                )

                # MIME Type
                doc_data["mimeType"] = fields.get("mimeType", {}).string_value or ""

            else:
                logger.warning(f"Document {doc.id} has no derived_struct_data or fields. Basic info only.")
                # Initialize fields to ensure consistent structure
                doc_data.setdefault("title", doc.name or doc.id) # Fallback title to document name/id
                doc_data.setdefault("author", "")
                doc_data.setdefault("date", "")
                doc_data.setdefault("full_content", "")
                doc_data.setdefault("link", "")
                doc_data.setdefault("mimeType", "")

            # Snippet Extraction (remains largely the same, might benefit from context of title/content)
            doc_data["snippet"] = ""
            if result.extractive_answers:
                 doc_data["snippet"] = result.extractive_answers[0].content
            elif response.summary and response.summary.summary_with_metadata:
                 doc_data["snippet"] = response.summary.summary_with_metadata.summary

            # If full_content is empty, snippet is better than nothing
            if not doc_data.get("full_content") and doc_data.get("snippet"):
                doc_data["full_content"] = doc_data["snippet"]

            # If title is still empty, try to use a snippet or part of content
            if not doc_data.get("title") and doc_data.get("full_content"):
                doc_data["title"] = (doc_data["full_content"][:100] + "...") if doc_data["full_content"] else doc.name or doc.id


            processed_results.append(doc_data)

        logger.info(f"Found {len(processed_results)} results.")
        return processed_results

    except GoogleAPIError as e:
        logger.error(f"Google API Error during Vertex AI Search query for datastore {datastore_path}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during Vertex AI Search query for datastore {datastore_path}: {e}", exc_info=True)
        return []

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

# --- Main Tool Function for Email Processing ---
def process_email_content_tool(
    email_documents: Union[Dict, List[Dict]],
    processing_level: str = "basic_clean",
    processing_options: Optional[Dict] = None
) -> Union[Dict, List[Dict]]:
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

    for doc_item in docs_to_process:
        if not isinstance(doc_item, dict) or 'full_content' not in doc_item:
            logger.warning(f"Skipping invalid document for email processing: {doc_item}")
            processed_docs_list.append(doc_item)
            continue

        current_text = str(doc_item.get('full_content', ''))
        processing_log: List[str] = []
        processed_doc = doc_item.copy()

        if not current_text.strip():
            processing_log.append("Original content was empty or whitespace.")
            processed_doc['cleaned_text'] = ""
            processed_doc['processing_log'] = processing_log
            processed_docs_list.append(processed_doc)
            continue

        if '<' in current_text and '>' in current_text and ('<html' in current_text.lower() or '<body' in current_text.lower() or '<p>' in current_text.lower()):
            text_before_html_strip = current_text
            current_text = _strip_html_bs(current_text)
            if text_before_html_strip != current_text:
                 processing_log.append("Stripped HTML content.")

        if processing_level == "basic_clean":
            text_before_replies = current_text
            current_text = _remove_quoted_replies(current_text)
            if text_before_replies != current_text:
                processing_log.append("Attempted to remove quoted replies.")

            if processing_options.get("remove_signatures", True):
                text_before_signatures = current_text
                custom_patterns = processing_options.get("signature_patterns")
                current_text = _remove_signatures(current_text, signature_patterns=custom_patterns)
                if text_before_signatures != current_text:
                    processing_log.append("Attempted to remove signatures.")

            text_before_whitespace = current_text
            current_text = _normalize_whitespace(current_text)
            if text_before_whitespace != current_text or not processing_log:
                processing_log.append("Normalized whitespace.")

            processed_doc['cleaned_text'] = current_text

        elif processing_level == "extract_conversation":
            logger.warning("Processing level 'extract_conversation' currently falls back to 'basic_clean'.")
            current_text = _remove_quoted_replies(current_text)
            if processing_options.get("remove_signatures", True):
                 current_text = _remove_signatures(current_text, signature_patterns=processing_options.get("signature_patterns"))
            current_text = _normalize_whitespace(current_text)
            processed_doc['cleaned_text'] = current_text
            processing_log.append("Performed basic_clean as fallback for extract_conversation.")

        else:
            logger.warning(f"Unsupported processing_level for email: {processing_level}. Returning original content.")
            processed_doc['cleaned_text'] = current_text
            processing_log.append(f"Unsupported processing_level for email: {processing_level}.")

        processed_doc['processing_log'] = processing_log
        processed_docs_list.append(processed_doc)

    return processed_docs_list[0] if is_single_doc else processed_docs_list

# --- Initial Version of process_drive_document_tool ---
def process_drive_document_tool(
    drive_document: Dict,
    processing_tasks: List[str],
    processing_options: Optional[Dict] = None
) -> Dict:
    """
    Processes a Google Drive document dictionary.
    Initial version focuses on tasks that can be done with content from Vertex AI Search
    or conceptually outlines where LLM/Drive API calls would be made.

    Args:
        drive_document: A single document dictionary from query_vertex_ai_search_tool.
                        Expected keys: 'id', 'name', 'mimeType', 'link', 'full_content', 'author', 'date'.
        processing_tasks: List of tasks to perform. Examples:
                          "summarize_text", "extract_keywords".
                          (Placeholders for "get_full_text", "convert_to_pdf_uri")
        processing_options: Options for specific tasks (e.g., {"summary_length": "short"}).

    Returns:
        The processed drive_document dictionary with new/augmented fields.
    """
    if processing_options is None:
        processing_options = {}

    logger.info(f"Executing process_drive_document_tool for doc ID: {drive_document.get('id')}, Tasks: {processing_tasks}")

    processed_doc = drive_document.copy()
    if 'processing_log' not in processed_doc:
        processed_doc['processing_log'] = []

    current_text_content = processed_doc.get('full_content', '')

    for task in processing_tasks:
        if task == "summarize_text":
            if current_text_content:
                processed_doc['summary'] = f"Summary of '{processed_doc.get('title', 'document')}' would appear here."
                processed_doc['processing_log'].append(f"Conceptual: Summarized text (length: {processing_options.get('summary_length', 'default')}).")
                logger.info(f"Task 'summarize_text' - conceptual LLM call for doc ID: {drive_document.get('id')}")
            else:
                processed_doc['processing_log'].append("Skipped summarize_text: No content available.")
                logger.warning(f"Task 'summarize_text' skipped for doc ID: {drive_document.get('id')} - no content.")

        elif task == "extract_keywords":
            if current_text_content:
                processed_doc['keywords'] = ["keyword1", "keyword2", "conceptual_keyword"]
                processed_doc['processing_log'].append("Conceptual: Extracted keywords.")
                logger.info(f"Task 'extract_keywords' - conceptual LLM call for doc ID: {drive_document.get('id')}")
            else:
                processed_doc['processing_log'].append("Skipped extract_keywords: No content available.")
                logger.warning(f"Task 'extract_keywords' skipped for doc ID: {drive_document.get('id')} - no content.")

        elif task == "get_full_text":
            processed_doc['processing_log'].append("Task 'get_full_text' is a placeholder for future Google Drive API integration.")
            logger.info(f"Task 'get_full_text' noted as placeholder for doc ID: {drive_document.get('id')}")

        elif task == "convert_to_pdf_uri":
            processed_doc['processing_log'].append("Task 'convert_to_pdf_uri' is a placeholder for future Google Drive API integration.")
            logger.info(f"Task 'convert_to_pdf_uri' noted as placeholder for doc ID: {drive_document.get('id')}")

        else:
            logger.warning(f"Unsupported processing task for Drive doc: {task} for doc ID: {drive_document.get('id')}")
            processed_doc['processing_log'].append(f"Unsupported task for Drive doc: {task}.")

    return processed_doc

# --- Initial Version of process_calendar_event_tool ---
def _parse_calendar_datetime_struct(datetime_struct: Optional[Dict]) -> Optional[str]:
    """Helper to parse Google Calendar API-like datetime objects (from JSON)."""
    if not datetime_struct or not isinstance(datetime_struct, dict):
        return None

    # Value might be directly in datetime_struct if it's already fields of startTime/endTime
    dt_value = datetime_struct.get('dateTime')
    date_value = datetime_struct.get('date')

    if dt_value and dt_value.get('stringValue'):
        try:
            dt_str = dt_value['stringValue']
            # Attempt to parse RFC3339 format
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %I:%M %p %Z') # Example: 2024-07-30 10:00 AM UTC
        except ValueError:
            logger.warning(f"Could not parse dateTime: {dt_str}")
            return dt_str # Return as is if parsing fails
    elif date_value and date_value.get('stringValue'): # For all-day events
        return date_value['stringValue'] + " (All day)"
    return None

def process_calendar_event_tool(
    calendar_event: Dict,
    processing_tasks: List[str],
    processing_options: Optional[Dict] = None
) -> Dict:
    """
    Processes a Google Calendar event dictionary.
    Initial version focuses on creating a human-readable summary.

    Args:
        calendar_event: A single event dictionary from query_vertex_ai_search_tool.
        processing_tasks: List of tasks. E.g., "format_for_display".
        processing_options: Options for specific tasks.

    Returns:
        The processed calendar_event dictionary with new/augmented fields.
    """
    if processing_options is None:
        processing_options = {}

    logger.info(f"Executing process_calendar_event_tool for event ID: {calendar_event.get('id')}, Tasks: {processing_tasks}")

    processed_event = calendar_event.copy()
    if 'processing_log' not in processed_event:
        processed_event['processing_log'] = []

    raw_fields = processed_event.get('derived_struct_data', {}).get('fields', {})

    for task in processing_tasks:
        if task == "format_for_display":
            summary_parts = []

            title = processed_event.get('title') or raw_fields.get('summary', {}).get('stringValue')
            if title:
                summary_parts.append(f"Event: {title}")
            else:
                summary_parts.append("Event: (No title)")

            start_time_str = None
            # Try to use 'date' field populated by query_vertex_ai_search_tool if it's a string
            if isinstance(processed_event.get('date'), str) and processed_event['date']:
                start_time_str = processed_event['date']

            # If not suitable, parse from raw 'startTime'
            if not start_time_str and raw_fields.get('startTime', {}).get('structValue', {}).get('fields'):
                 start_time_from_raw = _parse_calendar_datetime_struct(raw_fields.get('startTime').get('structValue').get('fields'))
                 if start_time_from_raw: start_time_str = start_time_from_raw


            end_time_str = None
            if raw_fields.get('endTime', {}).get('structValue', {}).get('fields'):
                end_time_from_raw = _parse_calendar_datetime_struct(raw_fields.get('endTime').get('structValue').get('fields'))
                if end_time_from_raw: end_time_str = end_time_from_raw


            if start_time_str:
                time_info = f"When: {start_time_str}"
                if end_time_str and end_time_str != start_time_str:
                    if "(All day)" in start_time_str and "(All day)" in end_time_str:
                         pass
                    elif "(All day)" in start_time_str:
                         pass
                    else:
                        time_info += f" to {end_time_str}"
                summary_parts.append(time_info)

            organizer = processed_event.get('author')
            if not organizer and raw_fields.get('organizer', {}).get('structValue', {}).get('fields', {}).get('email', {}).get('stringValue'):
                organizer = raw_fields.get('organizer').get('structValue').get('fields').get('email').get('stringValue')
            if organizer:
                summary_parts.append(f"Organizer: {organizer}")

            location = processed_event.get('location') or raw_fields.get('location', {}).get('stringValue')
            if location:
                summary_parts.append(f"Location: {location}")

            description = processed_event.get('full_content')
            if not description and raw_fields.get('description', {}).get('stringValue'):
                 description = raw_fields.get('description').get('stringValue')
            if description:
                summary_parts.append(f"Description: {description[:200] + '...' if len(description) > 200 else description}")

            attendees_list_val = raw_fields.get('attendees', {}).get('listValue', {}).get('values', [])
            if attendees_list_val: # Check if it's a non-empty list
                attendee_emails = []
                for att_val in attendees_list_val:
                    att_fields = att_val.get('structValue', {}).get('fields', {})
                    email = att_fields.get('email', {}).get('stringValue')
                    status = att_fields.get('responseStatus', {}).get('stringValue')
                    if email:
                        attendee_emails.append(f"{email} ({status})" if status else email)
                if attendee_emails: # Check if any emails were actually extracted
                    summary_parts.append(f"Attendees: {', '.join(attendee_emails)}")

            link = processed_event.get('link')
            if link:
                summary_parts.append(f"Link: {link}")

            processed_event['display_summary'] = ". ".join(summary_parts) + "."
            processed_event['processing_log'].append("Formatted event for display.")
            logger.info(f"Task 'format_for_display' completed for event ID: {calendar_event.get('id')}")

        elif task == "extract_action_items_from_description":
            description = processed_event.get('full_content', '')
            if description:
                processed_event['action_items'] = ["Conceptual action item 1", "Conceptual action item 2"]
                processed_event['processing_log'].append("Conceptual: Extracted action items from description.")
            else:
                processed_event['processing_log'].append("Skipped extract_action_items: No description available.")
        else:
            logger.warning(f"Unsupported processing task for Calendar event: {task} for event ID: {calendar_event.get('id')}")
            processed_event['processing_log'].append(f"Unsupported task for Calendar event: {task}.")

    return processed_event

# Example usage (for testing)
# if __name__ == '__main__':
#    logging.basicConfig(level=logging.INFO)

    # Drive example
    # sample_drive_doc = {
    #    "id": "drive_file_123",
    #    "title": "My Important Document",
    #    "mimeType": "application/vnd.google-apps.document",
    #    "link": "https://docs.google.com/document/d/drive_file_123/edit",
    #    "full_content": "This is the full text of my important document. It contains many details.",
    #    "author": "user@example.com",
    #    "date": "2024-01-15T10:00:00Z"
    # }
    # processed_drive = process_drive_document_tool(sample_drive_doc, ["summarize_text", "extract_keywords", "get_full_text"])
    # print("Processed Drive:", processed_drive)

    # Calendar example
    # sample_calendar_event = {
    #    "id": "cal_event_123",
    #    "title": "Team Strategy Meeting",
    #    "date": "2024-08-15T14:00:00Z",
    #    "author": "manager@example.com",
    #    "full_content": "Discuss Q4 strategy and roadmap. Bring ideas.",
    #    "link": "https://calendar.google.com/event?id=cal_event_123",
    #    "derived_struct_data": {
    #        "fields": {
    #            "startTime": {"structValue": {"fields": {"dateTime": {"stringValue": "2024-08-15T14:00:00Z"}}}},
    #            "endTime": {"structValue": {"fields": {"dateTime": {"stringValue": "2024-08-15T15:00:00Z"}}}},
    #            "organizer": {"structValue": {"fields": {"email": {"stringValue": "manager@example.com"}}}},
    #            "attendees": {"listValue": {"values": [
    #                {"structValue": {"fields": {"email": {"stringValue": "user1@example.com"}, "responseStatus": {"stringValue": "accepted"}}}},
    #                {"structValue": {"fields": {"email": {"stringValue": "user2@example.com"}, "responseStatus": {"stringValue": "tentative"}}}},
    #            ]}},
    #            "location": {"stringValue": "Meeting Room 3B"}
    #        }
    #    }
    # }
    # processed_event = process_calendar_event_tool(sample_calendar_event, ["format_for_display", "extract_action_items_from_description"])
    # print("\nProcessed Calendar Event Display:", processed_event.get("display_summary"))
    # print("Processed Calendar Event Action Items:", processed_event.get("action_items"))
    # print("Calendar Processing Log:", processed_event.get("processing_log"))
    # pass
