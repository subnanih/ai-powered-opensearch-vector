#!/bin/sh

set -e

echo "(Re)-creating directory"
rm -rf dependencies
mkdir dependencies
cd dependencies
echo "Downloading dependencies"
curl -sS https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip > sdk.zip
echo "Unpacking dependencies"

# (If you don't have `unzip` utility installed)
if command -v unzip &> /dev/null
then
    unzip sdk.zip && rm sdk.zip && echo "Done"
else
    echo "'unzip' command not found: Trying to unzip via Python"
    python -m zipfile -e sdk.zip . && rm sdk.zip && echo "Done"
fi
cd ..

# Install Python 3.9
sudo amazon-linux-extras install python3.9
sudo yum install -y python39-pip python39-devel

# Set Python 3.9 as default
sudo alternatives --set python3 /usr/bin/python3.9

# download requirements
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install --ignore-installed opensearch-py requests_aws4auth
python3 -m pip install pypdf==3.8.1 pydantic==1.10.8
python3 -m pip install sentence_transformers

python3 -m pip install --no-build-isolation --force-reinstall \
    "boto3>=1.28.57" \
    "awscli>=1.29.57" \
    "botocore>=1.31.57"

python3 -m pip install streamlit==1.27.0
