# tools/azure_search_builder.py
from promptflow import tool
from typing import Dict, Optional
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
import os
import logging

class SearchIndexBuilderTool:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.credential = DefaultAzureCredential()
        self.search_credential = AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
        self.search_client = SearchIndexClient(
            endpoint=f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net/",
            credential=self.search_credential
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )

    def create_search_resources(self, index_name: str, container_name: str) -> Dict:
        try:
            # Create data source
            datasource = self._create_blob_datasource(index_name, container_name)
            
            # Create index
            index = self._create_search_index(index_name)
            
            # Create indexer
            indexer = self._create_indexer(index_name, datasource.name)
            
            return {
                "status": "success",
                "index_name": index_name,
                "datasource_name": datasource.name,
                "indexer_name": indexer.name
            }
            
        except Exception as e:
            self.logger.error(f"Error creating search resources: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _create_blob_datasource(self, index_name: str, container_name: str):
        datasource = DataSource(
            name=f"{index_name}-datasource",
            type="azureblob",
            connection_string=self.blob_service_client.connection_string,
            container={"name": container_name}
        )
        return self.search_client.create_or_update_data_source(datasource)

    def _create_search_index(self, index_name: str):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="embedding_profile"
            ),
            SimpleField(
                name="category",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="sourcefile",
                type=SearchFieldDataType.String,
                filterable=True
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw_config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="embedding_profile",
                    algorithm_configuration_name="hnsw_config"
                )
            ]
        )

        semantic_config = SemanticConfiguration(
            name=f"{index_name}-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")]
            )
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=SemanticSettings(
                default_configuration=f"{index_name}-semantic-config",
                configurations=[semantic_config]
            )
        )
        
        return self.search_client.create_or_update_index(index)

    def _create_indexer(self, index_name: str, datasource_name: str):
        indexer = SearchIndexer(
            name=f"{index_name}-indexer",
            data_source_name=datasource_name,
            target_index_name=index_name,
            parameters=IndexingParameters(
                configuration={
                    "dataToExtract": "contentAndMetadata",
                    "parsingMode": "default",
                    "chunking": {
                        "mode": "document",
                        "maxChunkSize": 800,
                        "overlap": 100
                    }
                }
            ),
            field_mappings=[
                FieldMapping(source_field_name="metadata_storage_name", target_field_name="sourcefile"),
                FieldMapping(source_field_name="metadata_storage_file_extension", target_field_name="category")
            ]
        )
        return self.search_client.create_or_update_indexer(indexer)

@tool
def build_search_index(index_name: str, container_name: str) -> Dict:
    """
    Creates an Azure AI Search index for documents in a blob container with vector and semantic search capabilities.
    
    Args:
        index_name (str): Name of the search index to create
        container_name (str): Name of the blob container containing the documents
        
    Returns:
        Dict: Status and details of the created search resources
    """
    builder = SearchIndexBuilderTool()
    return builder.create_search_resources(index_name, container_name)
