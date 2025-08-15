import os
from llama_parse import LlamaParse
from core.config import settings
from core.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

def load_documents(data_dir: str = "data"):
    """
    Loads and parses all documents from the specified directory using LlamaParse.
    This version uses a simple synchronous loop for robustness and clear logging.
    """
    logger.info(f"Starting document loading from directory: {data_dir}")
    
    if not os.path.isdir(data_dir):
        logger.error(f"The path '{data_dir}' is not a valid directory. Please check the path.")
        return []

    # Initialize the LlamaParse parser
    parser = LlamaParse(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
    )

    # Find all files in the directory
    file_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames]
    
    if not file_paths:
        logger.warning(f"No files found in the directory: {data_dir}. Please add your documents.")
        return []

    logger.info(f"Found {len(file_paths)} file(s) to parse: {file_paths}")

    # Process each file one by one
    all_docs = []
    for file_path in file_paths:
        try:
            logger.info(f"--> Parsing file: {file_path}...")
            # Use the synchronous 'load_data' for simplicity
            parsed_file_docs = parser.load_data(file_path)
            
            if parsed_file_docs:
                logger.info(f"--> Successfully parsed {file_path}, found {len(parsed_file_docs)} document sections.")
                all_docs.extend(parsed_file_docs)
            else:
                logger.warning(f"--> Parsing {file_path} resulted in no documents.")

        except Exception as e:
            logger.error(f"--> Failed to parse file {file_path}. Error: {e}", exc_info=True)
    
    logger.info(f"\n✅ Total document sections parsed from all files: {len(all_docs)}")
    return all_docs  