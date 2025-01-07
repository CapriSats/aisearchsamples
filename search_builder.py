# search_builder.py
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
import os
import logging
from typing import Dict

class AzureSearchBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Azure clients with credentials from environment variables"""
        self.search_client = SearchIndexClient(
            endpoint=f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net/",
            credential=AzureKeyCredential(os.environ['AZURE_SEARCH_ADMIN_KEY'])
        )
        self.blob_client = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )

    def create_search_index(self, index_name: str, container_name: str) -> Dict:
        """
        Creates a search index and associated resources for the specified blob container.
        """
        try:
            # Create and configure the data source
            datasource = self._create_blob_datasource(index_name, container_name)
            
            # Create the search index with vector and semantic capabilities
            index = self._create_index_definition(index_name)
            self.search_client.create_or_update_index(index)
            
            # Create and start the indexer
            indexer = self._create_indexer_definition(index_name, datasource.name)
            self.search_client.create_or_update_indexer(indexer)
            
            return {
                "status": "success",
                "index_name": index_name,
                "endpoint": f"https://{os.environ['AZURE_SEARCH_SERVICE_NAME']}.search.windows.net/",
                "datasource": datasource.name,
                "indexer": indexer.name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create search resources: {str(e)}")
            raise RuntimeError(f"Search index creation failed: {str(e)}")

    def _create_blob_datasource(self, index_name: str, container_name: str):
        """Creates a blob storage data source connection"""
        datasource = DataSource(
            name=f"{index_name}-datasource",
            type="azureblob",
            connection_string=self.blob_client.connection_string,
            container={"name": container_name}
        )
        return self.search_client.create_or_update_data_source(datasource)

    def _create_index_definition(self, index_name: str) -> SearchIndex:
        """Creates the search index definition with vector and semantic search capabilities"""
        return SearchIndex(
            name=index_name,
            fields=[
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
                    vector_search_profile_name="default_vector_profile"
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
            ],
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default_hnsw",
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
                        name="default_vector_profile",
                        algorithm_configuration_name="default_hnsw"
                    )
                ]
            ),
            semantic_settings=SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name="default_semantic_config",
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="content")]
                        )
                    )
                ]
            )
        )

    def _create_indexer_definition(self, index_name: str, datasource_name: str) -> SearchIndexer:
        """Creates the indexer definition with document processing settings"""
        return SearchIndexer(
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
                FieldMapping(
                    source_field_name="metadata_storage_name",
                    target_field_name="sourcefile"
                ),
                FieldMapping(
                    source_field_name="metadata_storage_file_extension",
                    target_field_name="category"
                )
            ]
        )

def main(index_name: str, container_name: str) -> Dict:
    """
    Main entry point for creating an Azure AI Search index from a blob container.
    """
    builder = AzureSearchBuilder()
    return builder.create_search_index(index_name, container_name)

if __name__ == "__main__":
    # Example usage when running directly
    result = main("my-index", "my-container")
    print(result)
