from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.azure_aisearch import AzureAISearchVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
import os
import tempfile
from typing import List, Dict
import logging
from pathlib import Path

class LlamaIndexAzureSearch:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_clients()
        self._setup_llamaindex()

    def _initialize_clients(self):
        """Initialize Azure clients"""
        self.blob_client = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )
        self.search_credential = AzureKeyCredential(
            os.environ['AZURE_SEARCH_ADMIN_KEY']
        )

    def _setup_llamaindex(self):
        """Configure LlamaIndex with Azure OpenAI embeddings"""
        embedding_model = AzureOpenAIEmbedding(
            model=os.environ['AZURE_OPENAI_EMBEDDING_MODEL'],
            deployment_name=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        )
        
        Settings.embed_model = embedding_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 100

    def _download_blob_documents(self, container_name: str) -> List[Path]:
        """Download documents from blob storage to temporary directory"""
        container_client = self.blob_client.get_container_client(container_name)
        temp_dir = tempfile.mkdtemp()
        downloaded_files = []

        for blob in container_client.list_blobs():
            if blob.name.lower().endswith(('.pdf', '.docx', '.txt')):
                local_path = Path(temp_dir) / blob.name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                blob_client = container_client.get_blob_client(blob.name)
                with open(local_path, "wb") as file:
                    file.write(blob_client.download_blob().readall())
                
                downloaded_files.append(local_path)
                self.logger.info(f"Downloaded: {blob.name}")

        return downloaded_files

    def create_search_index(self, index_name: str, container_name: str) -> Dict:
        """Create and populate Azure AI Search index using LlamaIndex"""
        try:
            # Initialize Azure AI Search vector store
            vector_store = AzureAISearchVectorStore(
                search_endpoint=f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net",
                search_key=os.environ['AZURE_SEARCH_ADMIN_KEY'],
                index_name=index_name,
                dims=1536  # Dimensions for Azure OpenAI embeddings
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Download documents from blob storage
            document_paths = self._download_blob_documents(container_name)
            
            if not document_paths:
                raise ValueError("No valid documents found in blob container")

            # Create document nodes with metadata
            from llama_index.core import SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_files=document_paths
            ).load_data()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(Path(doc.metadata["file_path"]).name),
                    "container": container_name
                })

            # Create index with custom node parser
            node_parser = SentenceSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                node_parser=node_parser,
                show_progress=True
            )

            return {
                "status": "success",
                "index_name": index_name,
                "document_count": len(documents),
                "endpoint": f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net"
            }

        except Exception as e:
            self.logger.error(f"Failed to create search index: {str(e)}")
            raise RuntimeError(f"Index creation failed: {str(e)}")

        finally:
            # Cleanup temporary files
            for path in document_paths:
                try:
                    path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {path}: {str(e)}")

def main(index_name: str, container_name: str) -> Dict:
    """Main entry point for creating search index using LlamaIndex"""
    indexer = LlamaIndexAzureSearch()
    return indexer.create_search_index(index_name, container_name)

if __name__ == "__main__":
    # Example usage
    result = main("my-index", "my-container")
    print(result)
