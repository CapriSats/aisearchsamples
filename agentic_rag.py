# agentic_rag.py
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.azure_aisearch import AzureAISearchVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from azure.storage.blob import BlobServiceClient
from typing import Dict, List
import os
import tempfile
from pathlib import Path
import logging

class AgenticRAGSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize Azure services and LlamaIndex components"""
        self.blob_client = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )
        
        # Initialize Azure OpenAI
        self.llm = AzureOpenAI(
            model=os.environ['AZURE_OPENAI_MODEL'],
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        )
        
        # Initialize embeddings
        self.embed_model = AzureOpenAIEmbedding(
            model=os.environ['AZURE_OPENAI_EMBEDDING_MODEL'],
            deployment_name=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        )
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

    def process_documents(self, container_name: str) -> VectorStoreIndex:
        """Process documents from blob storage and create index"""
        temp_dir = tempfile.mkdtemp()
        downloaded_files = []
        
        try:
            container_client = self.blob_client.get_container_client(container_name)
            
            # Download and process documents
            for blob in container_client.list_blobs():
                if blob.name.lower().endswith(('.pdf', '.docx', '.txt')):
                    local_path = Path(temp_dir) / blob.name
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    blob_client = container_client.get_blob_client(blob.name)
                    with open(local_path, "wb") as file:
                        file.write(blob_client.download_blob().readall())
                    
                    downloaded_files.append(local_path)
            
            # Create vector store and index
            vector_store = AzureAISearchVectorStore(
                search_endpoint=f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net",
                search_key=os.environ['AZURE_SEARCH_ADMIN_KEY'],
                index_name=f"agentic-{container_name}",
                dims=1536
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load and process documents
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_files=downloaded_files
            ).load_data()
            
            # Create index with advanced configuration
            node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                node_parser=node_parser,
                show_progress=True
            )
            
            return index
            
        finally:
            # Cleanup temporary files
            for path in downloaded_files:
                try:
                    path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {path}: {str(e)}")

    def create_agentic_rag(self, container_name: str) -> Dict:
        """Create Agentic RAG system with the processed documents"""
        try:
            # Process documents and create index
            index = self.process_documents(container_name)
            
            # Create query engine with similarity post-processing
            query_engine = index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.7)
                ]
            )
            
            # Create agent with ReAct prompting
            agent = OpenAIAgent.from_tools(
                tools=[query_engine],
                llm=self.llm,
                chat_formatter=ReActChatFormatter(),
                verbose=True
            )
            
            return {
                "agent": agent,
                "index": index,
                "status": "success",
                "container": container_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create Agentic RAG system: {str(e)}")
            raise RuntimeError(f"Agentic RAG creation failed: {str(e)}")

def query_rag_system(agent: OpenAIAgent, query: str) -> str:
    """Execute a query against the Agentic RAG system"""
    try:
        response = agent.chat(query)
        return str(response)
    except Exception as e:
        logging.error(f"Query execution failed: {str(e)}")
        raise RuntimeError(f"Query failed: {str(e)}")

# Promptflow entry points
def initialize_rag(container_name: str) -> Dict:
    """Initialize the Agentic RAG system"""
    system = AgenticRAGSystem()
    return system.create_agentic_rag(container_name)

def query_system(agent: OpenAIAgent, query: str) -> str:
    """Query the Agentic RAG system"""
    return query_rag_system(agent, query)
