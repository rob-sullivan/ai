import requests
from bs4 import BeautifulSoup
import threading
from queue import Queue
import xml.etree.ElementTree as ET
from xml.dom import minidom
import signal
import sys
import csv

# Define the starting URL
start_url = 'https://revenue.ie/en/home.aspx'

# Define the file paths for visited URLs and URL queue
visited_urls_file = 'visited_urls.csv'
url_queue_file = 'url_queue.csv'

# Function to ensure the necessary files exist
def ensure_file_exists(file_path):
    open(file_path, 'a').close()

# Ensure the CSV files exist
ensure_file_exists(visited_urls_file)
ensure_file_exists(url_queue_file)

# Initialize a set for visited URLs and a queue for URLs to visit
visited_urls = set()
url_queue = Queue()

# Lock for thread-safe operations on the visited URLs set
visited_urls_lock = threading.Lock()

# Number of threads for concurrent crawling
num_threads = 4

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global stop_crawling
    print("Received CTRL+C. Stopping crawling...")
    stop_crawling = True

signal.signal(signal.SIGINT, signal_handler)

# Flag to control the crawling process
stop_crawling = False

# Function to crawl a single URL
def crawl_url(url):
    global visited_urls
    if stop_crawling:
        return

    print(f"Crawling: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            with visited_urls_lock:
                visited_urls.add(url)
                # Save the visited URL to a file
                with open(visited_urls_file, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([url])
            
            # Look for PDF links on the page
            for link in soup.find_all('a', href=True):
                if link['href'].endswith('.pdf'):
                    pdf_url = link['href']
                    print(f"\tFound PDF: {pdf_url}")
                    # You could add logic here to download the PDF or add it to a list for later processing
                    
                # Add new URLs to the queue (excluding PDFs if only interested in crawling pages)
                elif link['href'] not in visited_urls:
                    url_queue.put(link['href'])
                    with open(url_queue_file, 'a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([link['href']])
                    print(f"\tAdded URL to queue: {link['href']}")

    except Exception as e:
        print(f"Error crawling {url}: {e}")

# Function to start the crawling process
def start_crawling():
    while not stop_crawling:
        try:
            next_url = url_queue.get(timeout=1)
            crawl_url(next_url)
            url_queue.task_done()
        except Queue.Empty:
            continue

# Starting the crawling process
if __name__ == "__main__":
    # Load previously visited URLs and queue from files
    # (Skipping code for brevity, assume initialization as in the question)

    # Start crawler threads
    for _ in range(num_threads):
        t = threading.Thread(target=start_crawling)
        t.daemon = True
        t.start()

    # Add the starting URL to the queue
    url_queue.put(start_url)

    # Wait for the queue to be empty
    url_queue.join()

    print("Crawling complete!")