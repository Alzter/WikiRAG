import requests
import os, subprocess, json
import sys; sys.path.append("processor")
from wiki_extractor import WikiExtractor

class WikipediaDownload():
    """
    Downloads all of Wikipedia and converts it to raw text using [WikiExtractor](https://github.com/attardi/wikiextractor).
    """

    @staticmethod
    def download_wikipedia_dump(output_dir : str, download_subset = True, dump_url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'):
        """
        Downloads a dump of Wikipedia from https://dumps.wikimedia.org/ and saves it to directory ``output_dir``.

        Args:
            output_dir (str): The directory where the raw Wikipedia dump will be saved to.
            download_subset (bool): If true, only downloads a 5 MB subset of the Wikipedia dump.
            dump_url (str, optional): The URL of the Wikipedia dump file to be downloaded. Defaults to the latest English Wikipedia dump.
        
        Returns:
            dump_file_path (str): The path to the saved dump file.
        """

        dump_file_path = os.path.join(output_dir, 'wikipedia_dump_file.bz2')
        dump_folder = os.path.split(dump_file_path)[0]
        if not os.path.exists(dump_folder): os.makedirs(dump_folder)

        # Stream the file download based on user's choice (0 = full, 1 = subset)
        with requests.get(dump_url, stream=True) as r, open(dump_file_path, 'wb') as f:
            r.raise_for_status()  # Raise an error for bad responses
            
            # Download 5 MB subset
            if download_subset:
                downloaded_size = 0
                for chunk in r.iter_content(1024):  # Download data in chunks of 1 KB
                    if chunk:
                        f.write(chunk)  # Write the chunk to the file
                        downloaded_size += len(chunk)  # Increase size counter
                        if downloaded_size >= 5_000_000:  # Stop after 5 MB
                            break
            
            # Download the full dump
            else:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)
                print(f"Downloaded full dump.")
        
        print(f"Dump saved to {dump_file_path}")
        return dump_file_path

    @staticmethod
    def extract_wikipedia_dump(dump_file_path : str, output_dir='wikipedia_extracted', is_subset=True, use_local_wikiextractor = True):
        """
        Extracts plain text from the Wikipedia dump using WikiExtractor.

        Args:
            dump_file_path (str): The path to the downloaded Wikipedia dump file.
            output_dir (str, optional): The directory where the extracted text will be saved.
                                        Defaults to 'wikipedia_extracted'.
            is_subset (bool, optional): If True, assume the dump file is a small subset (e.g., 5MB).
                                        If False, process the full dump. Defaults to True.
            use_local_wikiextractor (bool, optional):

        
        Returns:
            output_dir (str): The location where the Wikipedia raw text dump is stored.
        """

        #  Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        if use_local_wikiextractor:
            if is_subset:
                WikiExtractor.main(input=dump_file_path, json=True, no_templates=True, output=output_dir, bytes="5M")
            else:
                WikiExtractor.main(input=dump_file_path, json=True, no_templates=True, output=output_dir)
        
        else:
            
            # Use '--bytes' for smaller subsets
            extractor_command = ['wikiextractor', '--json', '--no-templates', '-o', output_dir, dump_file_path]

            if is_subset:
                extractor_command.insert(1, '--bytes')
                extractor_command.insert(2, '5M')  #5  MB

            # Run wikiextractor via subprocess
            subprocess.run(extractor_command, check=True)

        print(f"Extraction completed.  Extracted files are saved in {output_dir}")

        return output_dir

    @staticmethod
    def download_and_extract_wikipedia_dump(output_dir, download_subset = False, dump_url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'):
        """
        Downloads a wikipedia dump and extracts the raw text from it.
        
        Args:
            output_dir (str): The directory where the extracted text will be saved.
                                        Defaults to 'wikipedia_extracted'.
            download_subset (bool, optional): If true, only downloads a 5 MB subset of the Wikipedia dump.
            dump_url (str, optional): The URL of the Wikipedia dump file to be downloaded.

        Returns:
            output_dir (str): The location where the Wikipedia raw text dump is stored.
        """
        # Download wikipedia dump archive in .bz2 format
        
        wikipedia_dump_path = WikipediaDownload.download_wikipedia_dump(output_dir, download_subset=download_subset, dump_url=dump_url)

        # Extract raw text from wikipedia dump
        wikipedia_output_dir = WikipediaDownload.extract_wikipedia_dump(wikipedia_dump_path, is_subset=download_subset, output_dir=output_dir, use_local_wikiextractor=True)

        # Delete original dump archive - we don't need it anymore
        os.remove(wikipedia_dump_path)

        return wikipedia_output_dir