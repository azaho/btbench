"""
    This script downloads all the files from the braintreebank.dev website.
    All the files are downloaded to the braintreebank/ directory.
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile

# Create braintreebank directory if it doesn't exist
if not os.path.exists('braintreebank_zip'):
    os.makedirs('braintreebank_zip')

# Get the main page
url = 'https://braintreebank.dev/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links on the page
links = soup.find_all('a')

# Download each linked file
for link in links:
    href = link.get('href')
    if href:
        # Make absolute URL if relative
        file_url = urljoin(url, href)
        
        # Get filename from URL
        filename = os.path.basename(file_url)
        
        if filename:  # Only proceed if there's a filename
            if (filename == 'brain_treebank_code_release') or (filename == '2411.08343'):
                continue

            filepath = os.path.join('braintreebank_zip', filename)
            
            # Check if file exists and is complete
            skip_download = False
            if os.path.exists(filepath):
                try:
                    # Get file size from server
                    response = requests.head(file_url)
                    expected_size = int(response.headers.get('content-length', 0))
                    
                    # Get local file size
                    actual_size = os.path.getsize(filepath)
                    
                    if expected_size == actual_size:
                        print(f'Skipping {filename} - already downloaded')
                        skip_download = True
                except:
                    # If any error occurs during size check, re-download to be safe
                    pass
            
            if skip_download:
                continue

            print(f'Downloading {filename}...')
            # Download the file with streaming to handle large files
            file_response = requests.get(file_url, stream=True)
            
            # Save to braintreebank directory
            with open(filepath, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f'Downloaded {filename}')