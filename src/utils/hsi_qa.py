import os
import logging
import json
import yaml
import time
import requests
import warnings
import pandas as pd
from tqdm import trange
from dotenv import load_dotenv
from io import BytesIO
import io
import zipfile
import re
import pickle

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
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
hf_key = os.getenv('HUGGINGFACE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
dense_embedder_api = os.getenv("HF_API_URL")

# Initialize clients
pc = Pinecone(api_key=pinecone_api_key)

# Define model
chat_model = "llama3-8b-8192"
index = pc.Index('hsi-notes')
namespace = 'Chapter-1'

class BM25Singleton:
    _instance = None

    @classmethod
    def get_instance(cls, texts=None):
        if cls._instance is None:
            if texts is None:
                raise ValueError("Initial texts required for the first initialization!")
            cls._instance = cls(texts)
        return cls._instance

    def __init__(self, texts):
        self.bm25 = BM25Encoder()
        self.bm25.fit(texts)

    def encode(self, queries):
        return self.bm25.encode_documents(queries)

class ExtractTextTableInfoFromPDF:
    def __init__(self, file_path):
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
                elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
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

def eval_table_index_llama(table_str):
    class Header(BaseModel):
        index: int = Field(description="Header of the table, 0 for first row as the header, 1 for first column as the header")
        
    parser = JsonOutputParser(pydantic_object=Header)

    chat = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    template = '''You will assist me in deciding, based on the first 2 entries of a table, whether the first row or the first colum should be the header. 
            You are to output an int, 0 or 1. Where 0 if the first row is header, and 1 if the first column is the header.
            Follow the format instructions carefully.
            Table:
            {table}
            
            {format_instructions}
            '''
    prompt = PromptTemplate(
        template=template,
        input_variables=["table"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | chat | parser
    return chain.invoke({"table": table_str})

def clean_values(x):
    if isinstance(x, str):
        return x.replace('_x000D_', '').strip()
    return x

def get_table_check_string(df):
    table_str = ""
    for i in range(2):
        if i ==1:
            table_str += f"Row {i}: {df.iloc[i].values.tolist()}"  
        else:
            table_str += f"Row {i}: {df.iloc[i].values.tolist()}\n"
    return table_str

def convert_table_to_str(df):
    for index, row in df.iterrows():
        row_str = ""
        for col in df.columns:
            sentences = re.split(r'(?<=\.)\s*', row[col])
            row_sentence = ""
            for i in range(len(sentences)):
                row_sentence += sentences[i] + "\n"
            row_str += f"{col}: {row_sentence}, "
        formatted = row_str[:-2]
    return formatted
    
def get_table_meta(elements):
    table_file_pages = {}
    for el in elements:
        # Using get to avoid KeyError and ensure 'filePaths' is not empty
        file_paths = el.get('filePaths')
        if file_paths:
            page = el.get('Page', 'Unknown')  # Provide a default page number if missing
            table_file_pages[file_paths[0]] = {"Page": page}
    return table_file_pages

def extract_data(extracted_data):
    if 'structuredData.json' in extracted_data:
        json_data = json.loads(extracted_data['structuredData.json'])
    return json_data
    
def get_table_pages_and_text_chunks(json_data):
    if 'elements' not in json_data:
        logging.error("Missing 'elements' key in json_data")
        raise ValueError("Missing 'elements' key in json_data")

    table_file_pages = {}
    page_text = ""
    start_page = 0
    all_chunks = []
    separator_list = ["\n\n", "\n", ". ", "!", "?", ",", " ", "", ")", "("]
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
                separators=separator_list)
    except Exception as e:
        logging.error(f"Failed to initialize text splitter: {e}")
        raise

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
                # Generate the chunks for the previous page
                separated_list = text_splitter.split_text(page_text)
                for chunk in separated_list:
                    if chunk not in [". ", "."]:  # Simplified condition
                        all_chunks.append({'ElementType': 'Text', 'Page': current_page, 'Text': chunk})
                # Update the string of text 
                page_text = ""
                list_label = ""
            else:
                if 'Text' in json_data['elements'][i]:  # Check if Text is not empty
                    if json_data['elements'][i]['Path'].endswith("Lbl") and not json_data['elements'][i]['Path'].startswith("//Document/Table"):
                        list_label = json_data['elements'][i]['Text']
                    else:
                        if list_label:
                            page_text += list_label + json_data['elements'][i]['Text']
                            list_label = ""  # Reset list label to empty string
                        else:
                            page_text += json_data['elements'][i]['Text'] + "\n"
        except KeyError as e:
            logging.warning(f"Key error in json_data['elements'][i] processing at index {i}: {e}")
    
    # Process the last page of the text
    if page_text:
        separated_list = text_splitter.split_text(page_text)
        for chunk in separated_list:
            if chunk not in [". ", "."]:
                all_chunks.append({'ElementType': 'Text', 'Page': start_page + 1, 'Text': chunk})
 
    # Obtaining the table metadata
    for i in range(len(json_data['elements'])):
        try:
            file_paths = json_data['elements'][i].get('filePaths')
            if file_paths:
                page = json_data['elements'][i].get('Page', 'Unknown')
                match = re.search(r'\d+', file_paths[0])
                table_index = match.group(0)
                table_file_pages[int(table_index)] = {"Page": page}
        except Exception as e:
            logging.error(f"Error processing file paths at index {i}: {e}")
    return table_file_pages, all_chunks

def get_tables(extractor):
    #The literal extraction of the file itself
    excel_files = {k: v for k, v in extractor.extracted_data.items() if k.endswith('.xlsx')}
    table_dataframes = {}

    num_tables =0
    for fname, content in excel_files.items():
        excel_stream = BytesIO(content)
        df = pd.read_excel(excel_stream, header=None)
        df = df.applymap(clean_values)
        df_str = get_table_check_string(df) 
        #dic = eval_table_index_llama(df_str)
        #header_index = dic['index']
        header_index = 1
        
        # If header_index is non zero
        if header_index:
            df = pd.read_excel(excel_stream, header=None)
            df = df.applymap(clean_values)
            df = df.T
            # Set the first row as the new header
            new_header = df.iloc[0]  # Take the first row for the header
            df = df[1:]  # Take the data less the header row
            df.columns = new_header  # Set the header row as the df header
            # Optionally, reset index if necessary
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.read_excel(excel_stream, header=0)
            
        table_str = convert_table_to_str(df)
        table_dataframes[num_tables] = table_str
        num_tables += 1
    return table_dataframes

def generate_tables(table_dataframes, table_file_pages):
    meta_table_batch = []
    table_dfs = []
    for table_index, table_str in table_dataframes.items():
        dic = {}
        dic['ElementType'] = 'Table'
        dic['Page'] = table_file_pages[table_index]['Page']
        dic['Table'] = table_dataframes[table_index]
        table_dfs.append(dic)

        meta_table_batch.append(f"ElementType 'Table', Page {table_file_pages[table_index]['Page']}, {table_dataframes[table_index]}")
    
    return table_dfs, meta_table_batch

def dense_embed(payload: str) -> str:
        response = requests.post(dense_embedder_api, headers={"Authorization": f"Bearer {hf_key}"}, json=payload)
        return response.json()

def prepare_pinecone_data(text_documents, table_dfs):
    text_df = pd.DataFrame(text_documents)
    tables_df = pd.DataFrame(table_dfs)
    all_texts = text_df['Text'].tolist() + tables_df['Table'].tolist()
    return all_texts, text_df, tables_df

# THIS DOES NOT INCLUDE THE ACTUAL UPSERT OF THE DATA ONTO PINECONE YET
def get_pinecone_data(text_df, tables_df, meta_table_batch, bm25):
        table_df_dict = tables_df.to_dict(orient="records") 

        # Section to embed and get out embeddings for the tables
        table_chunks = tables_df['Table'].tolist()
        table_sparse_embeddings = bm25.encode([combined for combined in meta_table_batch])
        table_dense_embeddings = dense_embed(table_chunks)
        
        if not isinstance(table_dense_embeddings, list):
            print("Embedding model not working properly")
            return None
        
            # Generate a list of IDs for the current batch
        table_ids = ['vec' +str(x) for x in range(len(meta_table_batch))]
        pinecone_table_upserts = []
            
        for _id, sparse, dense, meta in zip(table_ids, table_sparse_embeddings, table_dense_embeddings, table_df_dict):
                pinecone_table_upserts.append({
                    'id': _id,
                    'values': dense,
                    'sparse_values': sparse,
                    'metadata': meta
                })
        
        
        # Section to embed and get out embeddings for the Texts
        batched_pinecone_texts = []
        batch_size = 32
        for i in trange(0, len(text_df), batch_size):
            i_end = min(i+batch_size, len(text_df)) # Determine the end index of the current batch
            df_batch = text_df.iloc[i:i_end] # Extract the current batch from the DataFrame
            text_df_batch_dict = df_batch.to_dict(orient="records")

            meta_text_batch = [
                    f"ElementType 'Text', Page {row['Page']}: {row['Text']}" for _, row in text_df.iterrows()
                ]
            text_chunks = df_batch['Text'].tolist()
            text_sparse_embeddings = bm25.encode([combined for combined in meta_text_batch])
            text_dense_embeddings = dense_embed(text_chunks)
            if not isinstance(text_dense_embeddings, list):
                    print("Embedding model not working properly")
                    return None
                
            # Generate a list of IDs for the current batch
            text_ids = ['vec' +str(x) for x in range(i + len(table_ids), i_end)]
            pinecone_text_upserts = []
                
            for _id, sparse, dense, meta in zip(text_ids, text_sparse_embeddings, text_dense_embeddings, text_df_batch_dict):
                    pinecone_text_upserts.append({
                        'id': _id,
                        'values': dense,
                        'sparse_values': sparse,
                        'metadata': meta
                    })
            batched_pinecone_texts.append(pinecone_text_upserts)
        return batched_pinecone_texts, pinecone_table_upserts
    
# Function to upsert the pinecone texts and tables
def upsert_pinecone_data(pinecone_text, pinecone_tables):  
    def total_vector_count(pinecone_text, pinecone_tables):
        total = len(pinecone_tables)
        for batch in pinecone_text:
            total += len(batch)
        return total 
    
    def check_upsert_success(index, namespace):
        index_status = index.describe_index_stats()
        return index_status['namespaces'][namespace]['vector_count'] == total_vector_count(pinecone_text, pinecone_tables)
        
            
    # Upserting tables
    index_stats = pc.describe_index(os.environ['PINECONE_INDEX_NAME'])
    if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
        index.upsert(vectors = pinecone_tables, namespace=namespace)
        tables_upserted = True
    else:
        print("Pinecone database not ready. Check set up or connection... \n")
        return
    if tables_upserted:
        for batch_num in trange(len(pinecone_text)):
            if index_stats['status']['ready'] and index_stats['status']['state'] == "Ready":
                index.upsert(vectors = pinecone_text[batch_num], namespace=namespace)
            else:
                print("Pinecone database not ready. Check set up or connection... \n")
                return
            print(f"Batch {batch_num + 1} upserted")
            time.sleep(2)
        if check_upsert_success(index, namespace):
            return True
        else:
            print("Not all vectors were upserted. Exiting...")
            return
    else:
        print("Something went wrong while upserting the tables to the pinecone index. ")
    return

def save_bm25_instance(model_instance, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save bm25 model to use in future queries
    with open(model_path, 'wb') as file:
        pickle.dump(model_instance, file)
        
def upsert_all_pinecone_data(input_file_path):
    extractor = ExtractTextTableInfoFromPDF(input_file_path)
    extracted_data = extractor.extracted_data
    pdf_data = extract_data(extracted_data)
    table_file_pages, text_chunks = get_table_pages_and_text_chunks(pdf_data)
    table_dataframes = get_tables(extractor)
    table_dfs, meta_table_batch = generate_tables(table_dataframes, table_file_pages)
    all_texts, text_df, tables_df= prepare_pinecone_data(text_chunks, table_dfs)
    bm25_instance = BM25Singleton.get_instance(texts=all_texts)
    pinecone_text, pinecone_tables = get_pinecone_data(text_df, tables_df, meta_table_batch, bm25_instance)
    # Only uncomment if you want to upsert a new set of data
    # upsert_pinecone_data(pinecone_text, pinecone_tables)
    upsert_pinecone_data(pinecone_text, pinecone_tables)
    
    model_path = 'components/version1/bm25_model.pkl'
    save_bm25_instance(bm25_instance, model_path)
    return

