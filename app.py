from fastapi import FastAPI, HTTPException, Body
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
import pymongo
import requests
from bs4 import BeautifulSoup
import re
import os
from typing import Dict, Any, Optional, List
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get API keys and environment variables
def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key

def get_mongodb_uri():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI environment variable is not set")
        raise ValueError("MONGODB_URI environment variable is not set")
    return uri

# Setup MongoDB connection
def get_mongodb_client():
    try:
        mongodb_uri = get_mongodb_uri()
        client = pymongo.MongoClient(mongodb_uri)
        # Test the connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

# Initialize MongoDB Vector Store
def get_vector_store(collection_name="model_documentation"):
    try:
        client = get_mongodb_client()
        db = client["model_docs_db"]
        collection = db[collection_name]
        
        # Create a vector search index if it doesn't exist
        # Check if index exists
        indexes = collection.list_indexes()
        vector_index_exists = False
        for index in indexes:
            if index.get("name") == "vector_index":
                vector_index_exists = True
                break
        
        if not vector_index_exists:
            logger.info("Creating vector search index on MongoDB collection")
            collection.create_index(
                [("embedding", "vectorSearch")],
                name="vector_index",
                vectorSearchOptions={
                    "dimensions": 1536,  # Dimensionality of OpenAI embeddings
                    "similarity": "cosine"
                }
            )
        
        # Initialize the vector store
        embeddings = OpenAIEmbeddings(api_key=get_openai_api_key())
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding"
        )
        
        logger.info(f"Vector store initialized for collection: {collection_name}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

# Initialize OpenAI client with API key from environment
def get_llm():
    api_key = get_openai_api_key()
    return ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)

# Function to determine parent model from HF model page
def determine_parent_model(hf_model: str) -> str:
    url = f"https://huggingface.co/{hf_model}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.text.lower()
        
        # Define patterns to check for parent models
        parent_models = {
            "llama": ["llama", "meta-llama", "meta llama"],
            "llama2": ["llama-2", "llama 2"],
            "llama3": ["llama-3", "llama 3"],
            "mistral": ["mistral"],
            "mixtral": ["mixtral"],
            "gemini": ["gemini"],
            "mpt": ["mpt"],
            "imagegpt": ["imagegpt"],
            "phi": ["phi"],
            "falcon": ["falcon"],
            "gemma": ["gemma"],
            "gpt2": ["gpt2", "gpt-2"],
            "gpt-j": ["gpt-j"],
            "bert": ["bert"],
            "t5": ["t5"],
            "roberta": ["roberta"],
            "opt": ["opt"]
        }
        
        # Check for parent model mentions
        for model, patterns in parent_models.items():
            for pattern in patterns:
                if pattern in page_text:
                    # Special case for Llama versions
                    if model == "llama":
                        if "llama 3" in page_text or "llama-3" in page_text:
                            return "llama3"
                        elif "llama 2" in page_text or "llama-2" in page_text:
                            return "llama2"
                        else:
                            return "llama"
                    return model
        
        # Extract model architecture info using LLM
        llm = get_llm()
        schema = {
            "properties": {
                "model_architecture": {"type": "string", "description": "The parent model or architecture of this model"}
            },
            "required": ["model_architecture"]
        }
        
        extraction_chain = create_extraction_chain(schema, llm)
        extracted_data = extraction_chain.invoke(response.text)
        if extracted_data and "model_architecture" in extracted_data[0]:
            return extracted_data[0]["model_architecture"].lower()
            
        return "unknown"
    except Exception as e:
        logger.error(f"Error determining parent model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error determining parent model: {str(e)}")

# Function to get documentation URL for the parent model
def get_doc_url(parent_model: str) -> str:
    base_url = "https://huggingface.co/docs/transformers/model_doc/"
    
    # Handle special cases 
    model_mappings = {
        "llama": "llama",
        "llama2": "llama2",
        "llama3": "llama3",
        "mistral": "mistral",
        "mixtral": "mixtral", 
        "imagegpt": "imagegpt",
        "phi": "phi",
        "falcon": "falcon",
        "gemma": "gemma",
        "gpt2": "gpt2",
        "gpt-j": "gptj",
        "bert": "bert",
        "t5": "t5",
        "roberta": "roberta",
        "opt": "opt"
    }
    
    return f"{base_url}{model_mappings.get(parent_model, parent_model)}"

# Function to generate a unique ID for a document
def generate_doc_id(url: str, content_hash: str) -> str:
    return hashlib.md5(f"{url}:{content_hash}".encode()).hexdigest()

# Function to check if documentation is already stored in MongoDB
def documentation_exists(parent_model: str) -> bool:
    try:
        client = get_mongodb_client()
        db = client["model_docs_db"]
        collection = db["model_documentation"]
        
        # Check if documents for this model exist
        count = collection.count_documents({"metadata.parent_model": parent_model})
        return count > 0
    except Exception as e:
        logger.error(f"Error checking if documentation exists: {str(e)}")
        return False

# Function to fetch, process, and store documentation in MongoDB
def fetch_and_store_documentation(doc_url: str, parent_model: str) -> List[Document]:
    try:
        # Check if documentation is already stored
        if documentation_exists(parent_model):
            logger.info(f"Documentation for {parent_model} already exists in database")
            return retrieve_documentation_from_db(parent_model)
        
        logger.info(f"Fetching and storing documentation for {parent_model} from {doc_url}")
        
        # Step 1: Fetch the documentation
        loader = WebBaseLoader(doc_url)
        docs = loader.load()
        
        if not docs:
            logger.warning(f"No documents loaded from {doc_url}")
            return []
        
        # Step 2: Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"Split documentation into {len(split_docs)} chunks")
        
        # Step 3: Add metadata to documents
        for i, doc in enumerate(split_docs):
            # Create content hash
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            # Add metadata
            doc.metadata.update({
                "parent_model": parent_model,
                "source_url": doc_url,
                "chunk_id": i,
                "content_hash": content_hash,
                "doc_id": generate_doc_id(doc_url, content_hash)
            })
        
        # Step 4: Store documents in MongoDB Vector Store
        vector_store = get_vector_store()
        vector_store.add_documents(documents=split_docs)
        logger.info(f"Successfully stored {len(split_docs)} document chunks in MongoDB Atlas")
        
        return split_docs
    except Exception as e:
        logger.error(f"Error fetching and storing documentation: {str(e)}")
        raise

# Function to retrieve documentation from MongoDB
def retrieve_documentation_from_db(parent_model: str) -> List[Document]:
    try:
        client = get_mongodb_client()
        db = client["model_docs_db"]
        collection = db["model_documentation"]
        
        # Retrieve documents by parent_model
        cursor = collection.find({"metadata.parent_model": parent_model})
        
        docs = []
        for doc_data in cursor:
            doc = Document(
                page_content=doc_data.get("text", ""),
                metadata=doc_data.get("metadata", {})
            )
            docs.append(doc)
        
        logger.info(f"Retrieved {len(docs)} document chunks for {parent_model} from MongoDB")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documentation from MongoDB: {str(e)}")
        return []

# Function to search documentation using vector similarity
def search_documentation(query: str, parent_model: str, limit: int = 5) -> List[Document]:
    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(
            query=query,
            k=limit,
            pre_filter={"metadata.parent_model": parent_model}
        )
        
        logger.info(f"Found {len(docs)} relevant documents for query: {query}")
        return docs
    except Exception as e:
        logger.error(f"Error searching documentation: {str(e)}")
        return []

# Function to provide fallback requirements when all attempts to extract information fail
def get_fallback_requirements(parent_model: str) -> Dict[str, Any]:
    """Provides basic fallback requirements when all document retrieval methods have failed"""
    
    logger.warning(f"Using minimal fallback requirements for {parent_model} - this should be avoided")
    
    # Minimal fallback requirements - absolute basics only
    return {
        "required_params": ["pretrained_model_name_or_path"],
        "optional_params": {
            "temperature": "Controls randomness in generation"
        },
        "special_requirements": [],
        "methods": ["generate", "forward"],
        "input_format": {"text": "Input text to be processed"}
    }

# Method 1: Research via vector search in MongoDB if documentation is available
def research_via_vector_search(doc_url: str, parent_model: str) -> Dict[str, Any]:
    # Step 1: Fetch and store documentation if not already in database
    docs = fetch_and_store_documentation(doc_url, parent_model)
    
    if not docs:
        logger.warning(f"No documentation found for {parent_model} in vector store")
        raise ValueError("No documentation available in vector store")
    
    # Step 2: Search for relevant information in the documentation using targeted queries
    search_queries = [
        f"What are the required initialization parameters for {parent_model} model?",
        f"What optional parameters does the {parent_model} model support?",
        f"What are the special requirements for using {parent_model} model?",
        f"What methods are required for inference or text generation with {parent_model}?",
        f"How should inputs be formatted for {parent_model} model?"
    ]
    
    combined_requirements = {
        "required_params": [],
        "optional_params": {},
        "special_requirements": [],
        "methods": [],
        "input_format": {}
    }
    
    llm = get_llm()
    
    for query in search_queries:
        # Get relevant documents for the query
        relevant_docs = search_documentation(query, parent_model, limit=3)
        
        if not relevant_docs:
            continue
        
        # Combine document content
        doc_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Extract information using LLM with a targeted extraction prompt
        extraction_prompt = ChatPromptTemplate.from_template(
            """
            You are a model documentation specialist analyzing Hugging Face Transformer 
            documentation for the {parent_model} model. 
            
            Based on this documentation excerpt, please extract ONLY the factual information
            present in the text. Do NOT infer or guess information that isn't explicitly stated.
            
            Documentation excerpt:
            {text}
            
            Based ONLY on the information present in the text above, extract:
            1. Required parameters for model initialization that are EXPLICITLY mentioned
            2. Optional parameters that are EXPLICITLY mentioned
            3. Special requirements or considerations that are EXPLICITLY mentioned
            4. Required methods for inference or prediction that are EXPLICITLY mentioned
            5. Input formatting requirements that are EXPLICITLY mentioned
            
            If any category has no information in the excerpt, leave it empty rather than guessing.
            """
        )
        
        chain = extraction_prompt | llm
        result = chain.invoke({
            "parent_model": parent_model,
            "text": doc_text
        })
        
        # Process the LLM's response to extract structured requirements
        schema = {
            "properties": {
                "required_parameters": {"type": "array", "items": {"type": "string"}},
                "optional_parameters": {"type": "object"},
                "special_requirements": {"type": "array", "items": {"type": "string"}},
                "required_methods": {"type": "array", "items": {"type": "string"}},
                "input_format": {"type": "object"}
            }
        }
        
        extraction_chain = create_extraction_chain(schema, llm)
        extracted = extraction_chain.invoke(result.content)
        
        if extracted and len(extracted) > 0:
            extracted = extracted[0]
            if "required_parameters" in extracted:
                combined_requirements["required_params"].extend(extracted["required_parameters"])
            if "optional_parameters" in extracted:
                combined_requirements["optional_params"].update(extracted["optional_parameters"])
            if "special_requirements" in extracted:
                combined_requirements["special_requirements"].extend(extracted["special_requirements"])
            if "required_methods" in extracted:
                combined_requirements["methods"].extend(extracted["required_methods"])
            if "input_format" in extracted:
                combined_requirements["input_format"].update(extracted["input_format"])
    
    # Deduplicate the lists
    combined_requirements["required_params"] = list(set(combined_requirements["required_params"]))
    combined_requirements["special_requirements"] = list(set(combined_requirements["special_requirements"]))
    combined_requirements["methods"] = list(set(combined_requirements["methods"]))
    
    # Validate that we got meaningful results
    if not combined_requirements["required_params"]:
        raise ValueError("No required parameters found in vector search")
    
    return combined_requirements

# Method 2: Research via direct web scraping and analysis
def research_via_web(doc_url: str, parent_model: str) -> Dict[str, Any]:
    # Direct scraping of the documentation
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        
        # Load documentation
        loader = WebBaseLoader(doc_url)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No documents loaded from URL")
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Extract requirements from each chunk
        llm = get_llm()
        combined_requirements = {
            "required_params": [],
            "optional_params": {},
            "special_requirements": [],
            "methods": [],
            "input_format": {}
        }
        
        # Use a more directed prompt to extract accurate information
        extraction_prompt = ChatPromptTemplate.from_template(
            """
            You are analyzing official documentation for the {parent_model} model.
            
            Focus ONLY on extracting factual information that is explicitly stated in the text.
            Do NOT infer or guess any information that is not clearly stated.
            
            Documentation chunk:
            {text}
            
            Based ONLY on the information in this text, identify:
            1. Required parameters that are explicitly mentioned as necessary for model initialization
            2. Optional parameters that are explicitly mentioned
            3. Special requirements or considerations that are explicitly mentioned
            4. Methods that are explicitly mentioned as necessary for using this model
            5. Input formatting requirements that are explicitly mentioned
            
            Be precise and stick only to what is explicitly stated in the documentation.
            """
        )
        
        # Process limited number of chunks to avoid excessive API calls
        for doc in split_docs[:5]:
            chain = extraction_prompt | llm
            result = chain.invoke({
                "parent_model": parent_model,
                "text": doc.page_content
            })
            
            # Extract structured information
            schema = {
                "properties": {
                    "required_parameters": {"type": "array", "items": {"type": "string"}},
                    "optional_parameters": {"type": "object"},
                    "special_requirements": {"type": "array", "items": {"type": "string"}},
                    "required_methods": {"type": "array", "items": {"type": "string"}},
                    "input_format": {"type": "object"}
                }
            }
            
            extraction_chain = create_extraction_chain(schema, llm)
            extracted = extraction_chain.invoke(result.content)
            
            if extracted and len(extracted) > 0:
                extracted = extracted[0]
                if "required_parameters" in extracted:
                    combined_requirements["required_params"].extend(extracted["required_parameters"])
                if "optional_parameters" in extracted:
                    combined_requirements["optional_params"].update(extracted["optional_parameters"])
                if "special_requirements" in extracted:
                    combined_requirements["special_requirements"].extend(extracted["special_requirements"])
                if "required_methods" in extracted:
                    combined_requirements["methods"].extend(extracted["required_methods"])
                if "input_format" in extracted:
                    combined_requirements["input_format"].update(extracted["input_format"])
        
        # Deduplicate lists
        combined_requirements["required_params"] = list(set(combined_requirements["required_params"]))
        combined_requirements["special_requirements"] = list(set(combined_requirements["special_requirements"]))
        combined_requirements["methods"] = list(set(combined_requirements["methods"]))
        
        # If model name isn't in required_params, add it as it's almost always required
        model_param = "pretrained_model_name_or_path"
        if not combined_requirements["required_params"]:
            logger.warning(f"No required parameters found, adding {model_param} as basic requirement")
            combined_requirements["required_params"].append(model_param)
        
        return combined_requirements
        
    except Exception as e:
        logger.error(f"Error in research_via_web: {str(e)}")
        raise

# Method 3: Research via LLM analysis when other methods fail
def research_via_llm(doc_url: str, parent_model: str) -> Dict[str, Any]:
    llm = get_llm()
    
    # Have LLM analyze what the model likely requires based on its architecture
    research_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in Hugging Face models and the Transformers library.
        
        I need to understand the requirements for the {parent_model} model. Based on your knowledge
        of this model architecture and standard practices in the Transformers library, what are the
        likely requirements?
        
        Please be clear about what you know for certain versus what you're inferring based on
        similar models. Focus on:
        
        1. Required parameters for initializing this model
        2. Common optional parameters for this model type
        3. Special requirements or considerations
        4. Required methods for inference
        5. Input formatting requirements
        
        Documentation URL: {doc_url}
        
        Provide factual information where possible, and indicate your confidence level for any inferences.
        """
    )
    
    chain = research_prompt | llm
    result = chain.invoke({
        "parent_model": parent_model,
        "doc_url": doc_url
    })
    
    # Extract structured information from the LLM response
    schema = {
        "properties": {
            "required_parameters": {"type": "array", "items": {"type": "string"}},
            "optional_parameters": {"type": "object"},
            "special_requirements": {"type": "array", "items": {"type": "string"}},
            "required_methods": {"type": "array", "items": {"type": "string"}},
            "input_format": {"type": "object"},
            "confidence_level": {"type": "string", "description": "How confident the LLM is in this information"}
        }
    }
    
    extraction_chain = create_extraction_chain(schema, llm)
    extracted = extraction_chain.invoke(result.content)
    
    if not extracted or len(extracted) == 0:
        raise ValueError("Failed to extract requirements via LLM analysis")
    
    extracted = extracted[0]
    
    # Log confidence level
    if "confidence_level" in extracted:
        logger.info(f"LLM confidence level in {parent_model} requirements: {extracted['confidence_level']}")
    
    requirements = {
        "required_params": extracted.get("required_parameters", []),
        "optional_params": extracted.get("optional_parameters", {}),
        "special_requirements": extracted.get("special_requirements", []),
        "methods": extracted.get("required_methods", []),
        "input_format": extracted.get("input_format", {})
    }
    
    # Make sure we have at least the model path parameter
    if not requirements["required_params"]:
        requirements["required_params"].append("pretrained_model_name_or_path")
    
    return requirements

# Function to extract model requirements from stored documentation or research
def extract_model_requirements(doc_url: str, parent_model: str) -> Dict[str, Any]:
    try:
        logger.info(f"Researching requirements for {parent_model} model")
        
        # Multiple research methods in order of preference
        research_methods = [
            ("vector_search", lambda: research_via_vector_search(doc_url, parent_model)),
            ("web_research", lambda: research_via_web(doc_url, parent_model)),
            ("llm_analysis", lambda: research_via_llm(doc_url, parent_model))
        ]
        
        # Try each research method in order
        for method_name, research_func in research_methods:
            try:
                logger.info(f"Attempting to research via {method_name}")
                requirements = research_func()
                
                # Validate that we got meaningful results
                if (requirements and 
                    requirements.get("required_params") and 
                    len(requirements["required_params"]) > 0):
                    
                    logger.info(f"Successfully extracted requirements via {method_name}")
                    return requirements
                else:
                    logger.warning(f"{method_name} returned incomplete results, trying next method")
            except Exception as e:
                logger.warning(f"{method_name} failed: {str(e)}, trying next method")
        
        # If all research methods fail, return fallback requirements
        logger.error(f"All research methods failed for {parent_model}")
        return get_fallback_requirements(parent_model)
        
    except Exception as e:
        logger.error(f"Error in extract_model_requirements: {str(e)}")
        return get_fallback_requirements(parent_model)

# Function to check script against requirements and amend if needed
def validate_and_amend_script(script: str, requirements: Dict[str, Any], parent_model: str) -> str:
    # Use LLM to validate and amend the script
    validation_prompt = ChatPromptTemplate.from_template(
        """
        You are validating a predict.py script for a Hugging Face model. Your task is to check if the script meets 
        the model requirements and amend it if needed, preserving any valid content that might be for other purposes.
        
        Parent model: {parent_model}
        
        MODEL REQUIREMENTS:
        - Required parameters: {required_params}
        - Optional parameters: {optional_params}
        - Special requirements: {special_requirements}
        - Required methods: {methods}
        - Input format requirements: {input_format}
        
        Current script:
        ```python
        {script}
        ```
        
        Analyze the script and check that:
        1. All required parameters are in the script
        2. No invalid content is in the script for this model type
        3. The script follows the special requirements
        4. Required methods are implemented correctly
        5. Input format requirements are met
        
        Return the amended script, with necessary changes to make it valid according to the model requirements. 
        ONLY return the complete Python code without any explanations, comments, or placeholders.
        The code must be production-ready with no "TODO" comments or similar.
        """
    )
    
    # Format requirements for the prompt
    required_params_str = ", ".join(requirements["required_params"])
    optional_params_str = str(requirements["optional_params"])
    special_requirements_str = ", ".join(requirements["special_requirements"])
    methods_str = ", ".join(requirements["methods"])
    input_format_str = str(requirements["input_format"])
    
    # Get LLM with API key
    llm = get_llm()
    
    chain = validation_prompt | llm
    result = chain.invoke({
        "parent_model": parent_model,
        "required_params": required_params_str,
        "optional_params": optional_params_str,
        "special_requirements": special_requirements_str,
        "methods": methods_str,
        "input_format": input_format_str,
        "script": script
    })
    
    # Extract just the code from the response
    amended_script = result.content
    
    # If the response includes markdown code blocks, extract just the code
    if "```python" in amended_script:
        amended_script = re.search(r"```python\n(.*?)\n```", amended_script, re.DOTALL)
        if amended_script:
            amended_script = amended_script.group(1)
    elif "```" in amended_script:
        amended_script = re.search(r"```\n(.*?)\n```", amended_script, re.DOTALL)
        if amended_script:
            amended_script = amended_script.group(1)
    
    # Final validation of the amended script
    validation_result = validate_final_script(amended_script, requirements, parent_model)
    
    # If the script still has issues, attempt one more round of fixes
    if not validation_result["valid"]:
        fix_prompt = ChatPromptTemplate.from_template(
            """
            The amended script still has issues. Please fix these specific issues:
            
            {issues}
            
            Current script:
            ```python
            {script}
            ```
            
            Return only the fixed script without any explanations, comments, or placeholders.
            """
        )
        
        chain = fix_prompt | llm
        result = chain.invoke({
            "issues": ", ".join(validation_result["issues"]),
            "script": amended_script
        })
        
        amended_script = result.content
        
        # Extract just the code if needed
        if "```python" in amended_script:
            amended_script = re.search(r"```python\n(.*?)\n```", amended_script, re.DOTALL)
            if amended_script:
                amended_script = amended_script.group(1)
        elif "```" in amended_script:
            amended_script = re.search(r"```\n(.*?)\n```", amended_script, re.DOTALL)
            if amended_script:
                amended_script = amended_script.group(1)
    
    return amended_script

# Function to validate the final script
def validate_final_script(script: str, requirements: Dict[str, Any], parent_model: str) -> Dict[str, Any]:
    validation_prompt = ChatPromptTemplate.from_template(
        """
        You are validating a predict.py script for a {parent_model} model. Determine if the script meets all requirements.
        
        MODEL REQUIREMENTS:
        - Required parameters: {required_params}
        - Optional parameters: {optional_params}
        - Special requirements: {special_requirements}
        - Required methods: {methods}
        - Input format requirements: {input_format}
        
        Script to validate:
        ```python
        {script}
        ```
        
        Check that:
        1. All required parameters are in the script
        2. No invalid content is in the script for this model type
        3. The script follows the special requirements
        4. Required methods are implemented correctly
        5. Input format requirements are met
        6. The script is syntactically valid Python
        7. The script doesn't contain incomplete code, TODOs, or placeholders
        
        Respond with a JSON object with these fields:
        - valid: true or false
        - issues: a list of specific issues found (empty if valid)
        """
    )
    
    # Format requirements for the prompt
    required_params_str = ", ".join(requirements["required_params"])
    optional_params_str = str(requirements["optional_params"])
    special_requirements_str = ", ".join(requirements["special_requirements"])
    methods_str = ", ".join(requirements["methods"])
    input_format_str = str(requirements["input_format"])
    
    # Get LLM with API key
    llm = get_llm()
    
    chain = validation_prompt | llm
    result = chain.invoke({
        "parent_model": parent_model,
        "required_params": required_params_str,
        "optional_params": optional_params_str,
        "special_requirements": special_requirements_str,
        "methods": methods_str,
        "input_format": input_format_str,
        "script": script
    })
    
    # Extract JSON from response
    schema = {
        "properties": {
            "valid": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["valid", "issues"]
    }
    
    extraction_chain = create_extraction_chain(schema, llm)
    validation_result = extraction_chain.invoke(result.content)
    
    if validation_result and len(validation_result) > 0:
        return validation_result[0]
    else:
        return {"valid": False, "issues": ["Failed to validate script properly"]}

@app.get("/health")
async def health_check():
    # Check if API keys and connections are configured
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_status = "configured" if openai_key else "missing"
    
    mongodb_uri = os.environ.get("MONGODB_URI", "")
    mongodb_status = "configured" if mongodb_uri else "missing"
    
    # Test MongoDB connection if URI is provided
    mongo_connection = "untested"
    if mongodb_uri:
        try:
            client = pymongo.MongoClient(mongodb_uri)
            client.admin.command('ping')
            mongo_connection = "connected"
        except Exception as e:
            mongo_connection = f"failed: {str(e)}"
    
    return {
        "status": "healthy",
        "openai_api": openai_status,
        "mongodb_uri": mongodb_status,
        "mongodb_connection": mongo_connection
    }

@app.post("/validate")
async def validate_predict_script(
    request_data: Dict[str, str] = Body(...)
):
    try:
        hf_model = request_data.get("hf_model")
        script = request_data.get("script")
        
        if not hf_model or not script:
            raise HTTPException(status_code=400, detail="Both 'hf_model' and 'script' are required")
        
        # Step 1-2: Determine the parent model
        parent_model = determine_parent_model(hf_model)
        logger.info(f"Determined parent model: {parent_model}")
        
        # Step 3: Get documentation URL
        doc_url = get_doc_url(parent_model)
        logger.info(f"Using documentation URL: {doc_url}")
        
        # Step 4: Extract model requirements from documentation using vector store
        try:
            requirements = extract_model_requirements(doc_url, parent_model)
            logger.info(f"Extracted requirements: {requirements}")
        except Exception as e:
            logger.warning(f"Failed to extract requirements from documentation: {str(e)}")
            logger.info(f"Using fallback requirements for {parent_model}")
            requirements = get_fallback_requirements(parent_model)
        
        # Steps 5-8: Validate and amend the script
        amended_script = validate_and_amend_script(script, requirements, parent_model)
        
        # Return only the complete validated code
        return amended_script
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Use environment variable PORT for Cloud Run compatibility
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
