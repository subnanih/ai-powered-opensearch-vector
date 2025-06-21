from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json
import boto3
import os
import time
import sys
import signal

# Set the vector size for Titan Embeddings model
vector_size = 1536  # Amazon Titan Embeddings model dimension

# Initialize Bedrock client
region = os.environ.get('AOSS_VECTORSEARCH_REGION')
bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)

def generate_embedding(text):
    """Generate embeddings using Amazon Bedrock Titan Embeddings model"""
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            'inputText': text
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

# movies in JSON format
json_file_path = "sample-movies.json"

def full_load(index_name, client):
    # if index_name exists in collection, don't run this again 
    # create a new index
    if not client.indices.exists(index=index_name):
        print(f"Creating index '{index_name}'...")
        index_body = {
            "settings": {
                "index.knn": True
            },
            'mappings': {
                'properties': {
                    "title": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "v_title": { "type": "knn_vector", "dimension": vector_size },
                    "plot": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "v_plot": { "type": "knn_vector", "dimension": vector_size },
                    "actors": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "certificate": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "directors": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "genres": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "genres": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "image_url": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                    "gross_earning": {"type":"float"},
                    "metascore": {"type":"float"},
                    "rating": {"type":"double"},
                    "time_minute": {"type":"long"},
                    "vote": {"type":"long"},
                    "year": {"type":"long"}
                }
            }
        }

        client.indices.create(
            index=index_name, 
            body=index_body
        )
        print(f"Index '{index_name}' created successfully.")
        time.sleep(5)
    else:
        print(f"Index '{index_name}' already exists, continuing with data loading.")
    
    actions = []
    i = 0
    j = 0
    action = {"index": {"_index": index_name}}

    # Read and index the JSON data
    print("Starting to load data...")
    with open(json_file_path, 'r') as file:
        data = file.readlines()
        total_docs = len([line for line in data if not '"index"' in line])
        print(f"Found {total_docs} documents to process")
        
        for item in data:
            try:
                json_data = json.loads(item)
                if 'index' in json_data:
                    continue

                # Generate embedding for title using Bedrock
                title = json_data['title']
                v_title = generate_embedding(title)
                json_data['v_title'] = v_title
        
                if 'plot' in json_data:
                    # Generate embedding for plot using Bedrock
                    plot = json_data['plot']
                    v_plot = generate_embedding(plot)
                    json_data['v_plot'] = v_plot
        
                # Prepare bulk request
                actions.append(action)
                actions.append(json_data.copy())
        
                if i >= 10:
                    client.bulk(body=actions)
                    j += i
                    if j <= 500:  # Only show progress for first 500 documents
                        print(f"Processed {j}/{total_docs} documents ({(j/total_docs)*100:.1f}%)")
                    elif j % 100 == 0:  # After 500, only show occasional updates
                        print(f"Processed {j}/{total_docs} documents ({(j/total_docs)*100:.1f}%)")
                    i = 0
                    actions = []
                i += 1
            except Exception as e:
                print(f"Error processing document: {e}")
                continue

        # Send any remaining documents
        if actions:
            client.bulk(body=actions)
            j += i
            print(f"Processed {j}/{total_docs} documents ({(j/total_docs)*100:.1f}%)")

    print(f"\nData loading complete! {j} documents have been indexed.")
    print("The process will continue in the background.")
    print("You can now proceed with the next steps of the workshop.")

def main():
    host = os.environ.get('AOSS_VECTORSEARCH_ENDPOINT')
    region = os.environ.get('AOSS_VECTORSEARCH_REGION')
    index = "opensearch_movies"
    service = 'aoss'

    print(f"Starting data loading process to OpenSearch Serverless...")
    print(f"Host: {host}")
    print(f"Region: {region}")
    print(f"Index: {index}")

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                   session_token=credentials.token)

    # Build the OpenSearch client
    client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        timeout = 300,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    # Handle SIGINT (Ctrl+C) gracefully
    def signal_handler(sig, frame):
        print("\nProcess interrupted. Data loading will continue in the background.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print(f"OpenSearch Client - Sending to Amazon OpenSearch Serverless host {host} in Region {region}\n")
        
        # Process first 500 documents with detailed logging, then fork to background
        pid = os.fork()
        if pid == 0:  # Child process
            # Redirect stdout and stderr to /dev/null for the background process
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            full_load(index, client)
            sys.exit(0)
        else:  # Parent process
            # Wait for a short time to let the child process start
            time.sleep(1)
            print(f"Background process started with PID: {pid}")
            print("You can continue with the workshop while data loads in the background.")
            sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
