import os
import logging
import json
import uuid
import warnings
from tqdm import trange
from dotenv import load_dotenv
import io
import zipfile
import re
import time
import pickle
import nltk
from groq import Groq
from tqdm import tqdm
import sys

# Adobe PDF Services imports
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

# Pinecone and Langchain imports
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
hf_key = os.getenv('HUGGINGFACE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
dense_embedder_api = os.getenv("HF_API_URL")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHAT_MODEL   = "llama3-70b-8192"
client = Groq()
nltk.download('punkt_tab')

# Import other necessary modules
from llama_index.legacy import Document
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.vector_stores import PineconeVectorStore
from typing import Any, Callable, List, Optional, Sequence, TypedDict

import numpy as np

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.node_parser import NodeParser
from llama_index.legacy.node_parser.interface import NodeParser
from llama_index.legacy.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.legacy.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.legacy.schema import BaseNode, Document
from llama_index.legacy.utils import get_tqdm_iterable

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

import yaml
# Load the YAML file
with open('components/prompts.yaml', 'r') as file:
    config = yaml.safe_load(file)

RAG_GENERATE_ANSWER = config['RAG_GENERATE_ANSWER']
QUERY_REWRITER = config['QUERY_REWRITER']


# ## Utils
def is_file_present(folder_path, file_name):
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the path is a file
    return os.path.isfile(file_path)


# Initialize the logger
logging.basicConfig(level=logging.INFO)

class ExtractTextTableInfoFromPDF:
    def __init__(self, file_path):
        self.unique_id = str(uuid.uuid4())
        try:
            file = open(file_path, 'rb')
            input_stream = file.read()
            file.close()

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=os.getenv('ADOBE_SERVICES_CLIENT_ID'),
                client_secret=os.getenv('ADOBE_SERVICES_CLIENT_SECRET')
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT],
            )

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)
            
            zip_bytes = io.BytesIO(stream_asset.get_input_stream())
            with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
                # Extract all the contents into memory
                self.extracted_data = {name: zip_ref.read(name) for name in zip_ref.namelist()}
                
        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')
    
    # Generates a string containing a directory structure and file name for the output file using unique_id
    @staticmethod
    def create_output_file_path(unique_id: str) -> str:
        os.makedirs("../data/Extracted_data", exist_ok=True)
        return f"../data/Extracted_data/{unique_id}.zip"

    @classmethod
    def create_with_unique_id(cls, file_path):
        instance = cls(file_path)
        return instance, instance.unique_id

# Get the extracted data from the extractor
def get_extracted_data(extracted_data):
    if 'structuredData.json' in extracted_data:
        json_data = json.loads(extracted_data['structuredData.json'])
    return json_data

# Function to obtain text chunks using the text splitter
def get_text_chunks(file_path, json_data):
    if 'elements' not in json_data:
        logging.error("Missing 'elements' key in json_data")
        raise ValueError("Missing 'elements' key in json_data")
        
    file_name = file_path.split("/")[-1]

    # Chunks are split by pages here
    page_text = ""
    start_page = 0
    
    all_texts = []
    
    list_label = ""
    for i in range(len(json_data['elements'])):
        try:
            current_page = json_data['elements'][i]['Page']
        except KeyError:
            logging.warning(f"Missing 'Page' key in element at index {i}")
            continue

        try:
            if current_page > start_page:
                # Update the new page number
                start_page = current_page               
         
                all_texts.append({'ElementType': 'Text', 'file_name': file_name, 'Page': current_page, 'Text': page_text})
                page_text = ""
                list_label = ""
            else:
                if 'Text' in json_data['elements'][i]:  # Check if Text is not empty
                    if json_data['elements'][i]['Path'].endswith("Lbl") and not json_data['elements'][i]['Path'].startswith("//Document/Table"):
                        list_label = json_data['elements'][i]['Text']
                    else:
                        if list_label:
                            page_text += f"{list_label} {json_data['elements'][i]['Text']}\n"
                            list_label = ""  # Reset list label to empty string
                        else:
                            page_text += f"{json_data['elements'][i]['Text']}\n"
        except KeyError as e:
            logging.warning(f"Key error in json_data['elements'][i] processing at index {i}: {e}")
    
    
    # Processing the last page of the text
    if page_text:
        all_texts.append({'ElementType': 'Text', 'file_name': file_name, 'Page': current_page, 'Text': page_text})

    return all_texts

# Function to derive the nodes from the text chunks
def convert_pagetexts_to_nodes(text_chunks):
    
    # Function to clean up the text in each node
    def clean_up_text(content: str) -> str:
        """
        Remove unwanted characters and patterns in text input.
        :param content: Text input.
        :return: Cleaned version of original text input.
        """

        # Fix hyphenated words broken by newline
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

        # Remove specific unwanted patterns and characters
        unwanted_patterns = [
            "\\n", "  —", "——————————", "—————————", "—————",
            r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
        ]
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)

        # Fix improperly spaced hyphenated words and normalize whitespace
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)
        return content
    
    # Conversion of text chunks to Documents
    page_documents = [Document(text=chunk['Text'],
                            metadata={
                                "file_name": chunk['file_name'],
                                "page": chunk['Page']
                                },
                            excluded_llm_metadata_keys=["file_name"],
                            metadata_seperator="::",
                            metadata_template="{key}=>{value}",
                            text_template="Metadata: {metadata_str}\n-----\nContent: {content}")
                            
                            for chunk in text_chunks]

    # Clean the texts in each page
    page_nodes = []
    for d in page_documents:
        cleaned_text = clean_up_text(d.text)
        d.text = cleaned_text
        page_nodes.append(d)
    return page_nodes


def extract_pdf(file_path):
    extractor, unique_id = ExtractTextTableInfoFromPDF.create_with_unique_id(file_path)
    extracted_data = extractor.extracted_data
    pdf_data = get_extracted_data(extracted_data)
    filename = file_path.split("/")[-1]
    
    # Sent this information to database
    def export_to_db(fname):
        with open(f"../data/{fname}.json", "w", encoding="utf-8") as fout:
            json.dump(pdf_data, fout)
            
    export_to_db(filename)
    return unique_id

def load_pdf(file_path):
    fname = file_path.split("/")[-1]
    
    def load_from_db(fname):
        with open(f"../data/{fname}.json", "r", encoding='utf-8') as fin:
            pdf_data = json.load(fin)
        return pdf_data
    pdf_data = load_from_db(fname)
    return pdf_data


class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: List[float]

class SemanticSplitterNodeParser(NodeParser):
    """Semantic node parser.

    Splits a document into Nodes, with each node being a group of semantically related sentences.

    Args:
        buffer_size (int): number of sentences to group together when evaluating semantic similarity
        embed_model: (BaseEmbedding): embedding model to use
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )

    embed_model: BaseEmbedding = Field(
        description="The embedding model to use to for semantic comparison",
    )

    buffer_size: int = Field(
        default=1,
        description=(
            "The number of sentences to group together when evaluating semantic similarity. "
            "Set to 1 to consider each sentence individually. "
            "Set to >1 to group sentences together."
        ),
    )

    breakpoint_percentile_threshold = Field(
        default=95,
        description=(
            "The percentile of cosine dissimilarity that must be exceeded between a "
            "group of sentences and the next to form a node.  The smaller this "
            "number is, the more nodes will be generated"
        ),
    )

    @classmethod
    def class_name(cls) -> str:
        return "SemanticSplitterNodeParser"

    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SemanticSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        embed_model = embed_model or OpenAIEmbedding()

        id_func = id_func or default_id_func

        return cls(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_semantic_nodes_from_documents([node], show_progress)
            all_nodes.extend(nodes)

        return all_nodes

    def build_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text
            text_splits = self.sentence_splitter(text)

            sentences = self._build_sentence_groups(text_splits)

            combined_sentence_embeddings = self.embed_model.get_text_embedding_batch(
                [s["combined_sentence"] for s in sentences],
                show_progress=show_progress,
            )

            for i, embedding in enumerate(combined_sentence_embeddings):
                sentences[i]["combined_sentence_embedding"] = embedding

            distances = self._calculate_distances_between_sentence_groups(sentences)

            chunks = self._build_node_chunks(sentences, distances)

            nodes = build_nodes_from_splits(
                chunks,
                doc,
                id_func=self.id_func,
            )
            all_nodes.extend(nodes)

        return all_nodes

    def _build_sentence_groups(
        self, text_splits: List[str]
    ) -> List[SentenceCombination]:
        sentences: List[SentenceCombination] = [
            {
                "sentence": x,
                "index": i,
                "combined_sentence": "",
                "combined_sentence_embedding": [],
            }
            for i, x in enumerate(text_splits)
        ]

        # Group sentences and calculate embeddings for sentence groups
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]

            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_distances_between_sentence_groups(
        self, sentences: List[SentenceCombination]
    ) -> List[float]:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            similarity = self.embed_model.similarity(embedding_current, embedding_next)

            distance = 1 - similarity

            distances.append(distance)

        return distances

    def _build_node_chunks(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                end_index = index - 1

                group = sentences[start_index : end_index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks
    
class BM25Singleton:
    _instance = None
    @classmethod
    def get_instance(cls, texts=None):
        if cls._instance is None:
            if texts is None:
                raise ValueError("Initial texts required for the first initialization!")
            cls._instance = cls(texts)
        return cls._instance

    def __init__(self):
        self.bm25 = BM25Encoder()
        
    def fit(self, texts):
        self.bm25.fit(texts)

    def encode(self, queries):
        return self.bm25.encode_documents(queries)

def save_bm25_instance(model_instance, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save bm25 model to use in future queries
    with open(model_path, 'wb') as file:
        pickle.dump(model_instance, file)
    print("bm25 model saved\n")

def load_bm25_instance(pickle_path):
    with open(pickle_path, 'rb') as file:
        bm25_instance = pickle.load(file)
    print("bm25 model loaded")
    return bm25_instance

# Embedding model is customisable 
def load_embedding_model(model_name='sentence-transformers/all-mpnet-base-v2'):
    embed_model = HuggingFaceEmbedding(model_name)
    return embed_model


def get_semantic_nodes(embedding_model, page_documents, buffer_size=1, breakpoint_threshold=85):
    parser = SemanticSplitterNodeParser.from_defaults(
        embed_model = embedding_model,  
        buffer_size = buffer_size,  
        breakpoint_percentile_threshold = breakpoint_threshold,
        include_prev_next_rel = False
    )

    # Here we semantically chunk the nodes into semantically split nodes
    semantic_nodes = parser._parse_nodes(page_documents, show_progress=False)

    # proceed to embed each node so that u can upsert the lowercased text with the embeddings
    node_texts = [node.text.lower() for node in semantic_nodes]
    return node_texts


def fit_export_bm25(node_texts, bm_25_path):
    bm25_instance = BM25Singleton()
    # Fit the bm25 model on lowercased node texts
    bm25_instance.fit(texts=node_texts)
    
    save_bm25_instance(bm25_instance, model_path=bm_25_path)


def create_pinecone_index(hybrid_search):
    if index_name not in pc.list_indexes().names():
        logging.info("Creating pinecone index...")
        pc.create_index(
            index_name,
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logging.info(f"Pinecone index with name: \"{index_name}\" already created")

def generate_pinecone_upsert_data_batch(embedding_model, bm25_instance, node_texts, batch_size=30):
    pinecone_text_upserts = []

    for batch_start in tqdm(range(0, len(node_texts), batch_size), desc="Processing Batches"):
        batch_end = min(batch_start + batch_size, len(node_texts))
        batch_texts = node_texts[batch_start:batch_end]

        # Generate embeddings for the batch
        dense_embeddings = embedding_model._embed(batch_texts)
        # Generate sparse embeddings for the batch
        sparse_embeddings = bm25_instance.encode(batch_texts)

        for i in range(len(batch_texts)):
            pinecone_text_upserts.append({
                'id': f"vector{batch_start + i + 1}",
                'values': dense_embeddings[i],
                'sparse_values': sparse_embeddings[i],
                'metadata': {'text': batch_texts[i]}
            })

    return pinecone_text_upserts

def upsert_pinecone_data(file_path, pinecone_text_upserts, batch_size=30):  
    # Generate a new UUID for the namespace
    namespace = file_path.split("/")[-1]
    logging.info(f"Starting upsertion to namespace {namespace}...")

    # Upsert data in batches
    for batch_start in tqdm(range(0, len(pinecone_text_upserts), batch_size), desc="Upserting Batches"):
        batch_end = min(batch_start + batch_size, len(pinecone_text_upserts))
        pinecone_batch = pinecone_text_upserts[batch_start:batch_end]

        pinecone_index.upsert(vectors=pinecone_batch, namespace=namespace)
        logging.info(f"Upserting batch {batch_start // batch_size + 1}...")

    time.sleep(10)
    index_status = pinecone_index.describe_index_stats()
    time.sleep(5)

    if index_status['namespaces'][namespace]['vector_count'] == len(pinecone_text_upserts):
        logging.info(f"All vectors uploaded successfully to namespace {namespace}")
        return namespace, True
    else:
        logging.error(f"Not all vectors were upserted to namespace {namespace}. Exiting...")
        return namespace, False


def modify_vector_query(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

# Usage of hybrid vector normaliser
# hdense, hsparse = modify_vector_query(dense_vector, sparse_vector, alpha=0.75)

def retrieve_context(index, index_name, namespace, query, embedding_model, bm25_model, top_k ):
    index_stats = pc.describe_index(index_name)
    if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
        dense_query = embedding_model._embed(query)
        sparse_query = bm25_model.encode(query)
        relevant_matches = index.query( 
            namespace=namespace,
            top_k=top_k, 
            vector=dense_query, 
            sparse_vector=sparse_query, 
            include_metadata=True
            )
        
        processed_context = [{'vector': result['id'], 'text':result['metadata']['text'], 'retrieval_score':result['score']} for result in relevant_matches['matches']]
        return processed_context
    else:
        logging.error("Pinecone index not ready for retrieval, check connection properly...")

# Edit this for query re writing function
def enhance_query(query, query_rewriter_prompt):
    # Prepare the prompt using the provided answer prompt template, text, and list of questions
    prompt = PromptTemplate(
        template=query_rewriter_prompt,
        input_variables=["user_query"],
    ) 
    # Format the final prompt with the actual text data and question list
    final_prompt = prompt.format(user_query=query.lower())
    # Generate the completion by interacting with the language model API
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0,  # Control the randomness of the output (lower means less random)
        max_tokens=1024,  # Limit the response length
        top_p=1,  # Nucleus sampling parameter (1 means only the most likely tokens are considered)
        stream=True,  # Enable streaming of the response chunks
        stop=None,  # Define stopping conditions (None means no stopping condition)
    )

    # Initialize an empty string to accumulate the response content
    answer = ''''''
    for chunk in completion:
        # Append each chunk of content to the answer string
        answer += chunk.choices[0].delta.content or ""
    cleaned_answer = extract_answer(answer, pattern=r'{"Query":\s*".*?"}')
    # Return the dictionary containing the generated answers
    return cleaned_answer

def print_context(context):
    for con in context:
        print("Retrieved context:\n")
        print(con['text'] + "\n")
        print(f"Similarity Score: {con['retrieval_score']}")
        print("-" * 80)
    
def extract_context(context, top_k):
    new_context = ''''''
    for i in range(len(context)):
        new_context += context[i]['text']
        if i == top_k:
            break   
    return new_context
        
def extract_answer(input_string, pattern=None):
    # Find the start and end indices of the JSON data within the input string
    # Assuming the JSON data starts with '{' and ends with '}'
    json_start = input_string.find('{')
    json_end = input_string.rfind('}') + 1
    
    # If either the start or end index is not found, raise an error
    if json_start == -1 or json_end == -1:
        raise ValueError("Invalid input: No JSON data found.")

    # Extract the substring that potentially contains the JSON data
    json_data = input_string[json_start:json_end]
    
    try:
        # Attempt to convert the JSON string to a Python dictionary
        data_dict = json.loads(json_data)
        return data_dict
    
    except json.JSONDecodeError:
        # If JSON decoding fails, use the provided pattern or default pattern to search for a JSON object
        if not pattern:
            # Default pattern matches JSON objects containing the 'Answer' key
            pattern = r'{"Answer":\s*".*?"}'
        
        match = re.search(pattern, input_string, re.DOTALL)

        if match:
            # If a match is found, extract the matched JSON string and convert it to a dictionary
            data_json_str = match.group()
            data_dict = json.loads(data_json_str)
            return data_dict

        # If no valid JSON is found, the function will log an error
        else:
            print(input_string)
            logging.error("No dictionary with the specified pattern found in this input string. Error by LLM")
            return {"error": "No dictionary with the specified pattern found"}

def rag_chat(question, context, rag_prompt, client):
    # Prepare the prompt using the provided answer prompt template, text, and list of questions
    prompt = PromptTemplate(
        template=rag_prompt,
        input_variables=["context", "question"],
    ) 
    
    # Format the final prompt with the actual text data and question list
    final_prompt = prompt.format(context=context, question=question)
    # Generate the completion by interacting with the language model API
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0,  # Control the randomness of the output (lower means less random)
        max_tokens=1024,  # Limit the response length
        top_p=1,  # Nucleus sampling parameter (1 means only the most likely tokens are considered)
        stream=True,  # Enable streaming of the response chunks
        stop=None,  # Define stopping conditions (None means no stopping condition)
    )

    # Initialize an empty string to accumulate the response content
    answer = ''''''
    for chunk in completion:
        # Append each chunk of content to the answer string
        answer += chunk.choices[0].delta.content or ""
    cleaned_answer = extract_answer(answer, pattern=r'{"Answer":\s*".*?"}')
    # Return the dictionary containing the generated answers
    return cleaned_answer


def rag_pipeline(query, rag_prompt, query_rewriter_prompt, top_k):
    # Function to send query to database
    enhanced_query = enhance_query(query, query_rewriter_prompt)['Query']
    retrieved_context = retrieve_context(pinecone_index, index_name, namespace, enhanced_query, embed_model, bm25_instance, top_k=5)
    extracted_context = extract_context(retrieved_context, top_k)
    answer = rag_chat(enhanced_query, extracted_context, rag_prompt, client)['Answer']
    return answer

def chatbot():
    i=0
    while True:
        # Take user input
        if i>0:
            user_query = input("Your next question: ")
   
        user_query = input("Type in your question here: ")

        # Exit the loop if the user wants to quit
        if user_query.lower() in ['exit', 'Exit']:
            print("Thanks for chatting, hope this was helpful!")
            break

        # Process the query with your RAG system
        answer = pipeline(user_query, RAG_GENERATE_ANSWER, top_k=5)

        print("-" * 100)
        print("ANSWER:\n\n" + answer + "\n")
        print("-" * 100)
        i += 1
        

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        # Convert the first argument to a boolean based on "yes" or "no"
        input_bool = sys.argv[1].lower()
        if input_bool in ['yes', 'y']:
            to_upsert = True
        elif input_bool in ['no', 'n']:
            to_upsert = False
        else:
            logging.error("Invalid input: please only input 'yes' or 'no'.")
            sys.exit(1)  # Exit the script with an error code
    else:
        # No input provided, assume no upsert is needed
        to_upsert = False
        
    file_path = '../PDF/HSI1000-chapter1.pdf'
    filename = file_path.split("/")[-1]
    pickle_path = f'../src/components/hybrid-rag/{filename}bm25_model.pkl'
    namespace = filename
    
    if to_upsert:

        # extract_pdf(file_path)
        pdf_data = load_pdf(file_path)
        page_texts = get_text_chunks(file_path, pdf_data)
        page_documents = convert_pagetexts_to_nodes(page_texts)

        embed_model = load_embedding_model()

        # Limit the number of page_documents to embed
        node_texts = get_semantic_nodes(embed_model, page_documents[:3])

        if not is_file_present(folder_path = "../src/components/hybrid-rag", file_name = f"{filename}bm25_model.pkl"):
            fit_export_bm25(node_texts, bm_25_path=pickle_path)
            
        bm25_instance = load_bm25_instance(pickle_path=pickle_path)
        pinecone_data = generate_pinecone_upsert_data_batch(embed_model, bm25_instance, node_texts)

        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "rag-testing-hybrid"
        create_pinecone_index(hybrid_search=True)

        # Initialize your index
        pinecone_index = pc.Index(index_name)

        namespace, success = upsert_pinecone_data(file_path, pinecone_data)
        if success:
            logging.info(f"Data successfully upserted into namespace: {namespace}")
        else:
            logging.error(f"Failed to upsert data into namespace: {namespace}")
    else:
        embed_model = load_embedding_model()
        bm25_instance = load_bm25_instance(pickle_path=pickle_path)
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "rag-testing-hybrid"
        # Initialize your index
        pinecone_index = pc.Index(index_name)

        
        # some queries to try:
        # query = "What is the scientific method used for?"
        # query = "Why is science self correcting?"
        # query = "Why is science self-correcting?"

        # Chatbot function
    i=0
    while True:
            # Take user input
            if i>0:
                user_query = input("Your next question: ")
            else:
                user_query = input("Type in your question here: ")

            # Exit the loop if the user wants to quit
            if user_query.lower() in ['exit', 'Exit']:
                print("Thanks for chatting, hope this was helpful!")
                break
            
            # Process the query with your RAG system
            answer = rag_pipeline(user_query, RAG_GENERATE_ANSWER, QUERY_REWRITER, top_k=5)

            print("-" * 100)
            print("ANSWER:\n\n" + answer + "\n")
            print("-" * 100)
            i += 1
