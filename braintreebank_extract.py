"""
Extract zip files in braintreebank_zip directory to braintreebank directory
"""

import os
import zipfile
# Create braintreebank directory if it doesn't exist
if not os.path.exists('braintreebank'):
    os.makedirs('braintreebank')

# Extract zip files
successful = 0
failed = 0
for filename in os.listdir('braintreebank_zip'):
    if filename.endswith('.zip'):
        print(f'Extracting {filename}...')
        try:
            with zipfile.ZipFile(os.path.join('braintreebank_zip', filename), 'r') as zip_ref:
                zip_ref.extractall('braintreebank')
            print(f'Done.')
            successful += 1
        except:
            print(f'Failed to extract.')
            failed += 1

print(f'\nExtraction complete: {successful} files extracted successfully, {failed} files failed')