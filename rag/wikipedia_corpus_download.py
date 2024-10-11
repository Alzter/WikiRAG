import requests
import os, subprocess, json
import sys;sys.path.append("rag")
from wiki_extractor import WikiExtractor

class WikipediaDownload():
    """
    Downloads all of Wikipedia and converts it to raw text using [WikiExtractor](https://github.com/attardi/wikiextractor).
    """

    @staticmethod
    def download_wikipedia_dump(output_dir : str, max_megabytes : int | None = None, dump_url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'):
        """
        Downloads a dump of Wikipedia from https://dumps.wikimedia.org/ and saves it to directory ``output_dir``.

        Args:
            output_dir (str): The directory where the raw Wikipedia dump will be saved to.
            max_megabytes (int, optional): If true, only downloads ``max_megabytes`` MB worth of data from the Wikipedia dump.
            dump_url (str, optional): The URL of the Wikipedia dump file to be downloaded. Defaults to the latest English Wikipedia dump.
        
        Returns:
            dump_file_path (str): The path to the saved dump file.
        """

        dump_file_path = os.path.join(output_dir, 'wikipedia_dump_file.bz2')
        dump_folder = os.path.split(dump_file_path)[0]
        if not os.path.exists(dump_folder): os.makedirs(dump_folder)

        # Stream the file download
        with requests.get(dump_url, stream=True) as r, open(dump_file_path, 'wb') as f:

            r.raise_for_status()  # Raise an error for bad responses
            
            downloaded_size = 0
            for chunk in r.iter_content(1024):  # Download data in chunks of 1 KB
                if chunk:
                    f.write(chunk)  # Write the chunk to the file
                    downloaded_size += len(chunk)  # Increase size counter

                    if max_megabytes is not None:
                        max_bytes = max_megabytes * 1_000_000
                        if downloaded_size >= max_bytes:
                            break
            
            if max_megabytes is not None:
                print(f"Downloaded {max_megabytes}MB subset of dump.")
            else: print(f"Downloaded full dump.")
        
        print(f"Dump saved to {dump_file_path}")
        return dump_file_path

    @staticmethod
    def extract_wikipedia_dump(dump_file_path : str, output_dir='wikipedia_extracted', subfile_max_megabytes : int = 10, use_local_wikiextractor = True):
        """
        Extracts plain text from the Wikipedia dump using WikiExtractor and saves into ``output_dir`` as multiple text files.
        Each text file will contain ``subfile_max_megabytes`` MB worth of Wikipedia articles.

        Args:
            dump_file_path (str): The path to the downloaded Wikipedia dump file.
            output_dir (str, optional):             The directory where the extracted text will be saved.
                                                    Defaults to 'wikipedia_extracted'.
            subfile_max_megabytes (int, optional): The maximum filesize for each raw text file. The higher this number, the less text files will be stored.
            use_local_wikiextractor (bool, optional):

        
        Returns:
            output_dir (str): The location where the Wikipedia raw text dump is stored.
        """

        #  Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        subfile_bytes = f"{subfile_max_megabytes}M"

        if use_local_wikiextractor:
            WikiExtractor.main(input=dump_file_path, json=True, no_templates=True, output=output_dir, bytes=subfile_bytes)
        
        else:
            
            # Use '--bytes' for smaller subsets
            extractor_command = ['wikiextractor', '--bytes', subfile_bytes, '--json', '--no-templates', '-o', output_dir, dump_file_path]

            # if subset_size_megabytes is not None:
            #     extractor_command.insert(1, '--bytes')
            #     extractor_command.insert(2, f'{ str(subset_size_megabytes) }M')  #5  MB

            # Run wikiextractor via subprocess
            subprocess.run(extractor_command, check=True)

        print(f"Extraction completed.  Extracted files are saved in {output_dir}")

        return output_dir

    @staticmethod
    def download_and_extract_wikipedia_dump(output_dir, subfile_max_megabytes : int = 5, max_megabytes : int | None = None, dump_url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'):
        """
        Downloads a wikipedia dump and extracts the raw text from it.
        
        Args:
            output_dir (str): The directory where the extracted text will be saved.
                                        Defaults to 'wikipedia_extracted'.
            subfile_max_megabytes (int, optional): The maximum filesize for each raw text file. The higher this number, the less text files will be stored.
            max_megabytes (int, optional): If true, only downloads ``max_megabytes`` MB worth of data from the Wikipedia dump.
            dump_url (str, optional): The URL of the Wikipedia dump file to be downloaded.

        Returns:
            output_dir (str): The location where the Wikipedia raw text dump is stored.
        """
        # Download wikipedia dump archive in .bz2 format
        
        wikipedia_dump_path = WikipediaDownload.download_wikipedia_dump(output_dir, max_megabytes=max_megabytes, dump_url=dump_url)

        # Extract raw text from wikipedia dump
        wikipedia_output_dir = WikipediaDownload.extract_wikipedia_dump(wikipedia_dump_path, output_dir=output_dir, subfile_max_megabytes=subfile_max_megabytes, use_local_wikiextractor=True)

        # Delete original dump archive - we don't need it anymore
        os.remove(wikipedia_dump_path)

        return wikipedia_output_dir