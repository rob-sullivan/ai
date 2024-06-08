import requests
from bs4 import BeautifulSoup
import threading
from queue import Queue, Empty
import xml.etree.ElementTree as ET
from xml.dom import minidom
import signal
import sys
import csv
from urllib.parse import urlparse, urljoin

# Define the starting URL
start_url = 'https://revenue.ie/en/home.aspx'

def ensure_file_exists(file_path):
    open(file_path, 'a').close()

ensure_file_exists('visited_urls.csv')
ensure_file_exists('url_queue.csv')

# Initialize a set to keep track of visited URLs
visited_urls_file = 'visited_urls.csv'
try:
    with open(visited_urls_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        visited_urls = {rows[0] for rows in reader}
except FileNotFoundError:
    visited_urls = set()

url_queue_file = 'url_queue.csv'
# Initialize a queue for URLs to visit
url_queue = Queue()

# Try to load URL queue from a file
try:
    with open(url_queue_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            url_queue.put(row[0])
except FileNotFoundError:
    pass  # If the file doesn't exist, start with the initial URL

# Lock to synchronize access to the visited_urls set
visited_urls_lock = threading.Lock()

# Number of crawler threads
num_threads = 4

# Create an XML sitemap
sitemap = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

# Flag to indicate when to stop crawling
stop_crawling = False

# Signal handler to stop crawling on CTRL+C
def signal_handler(sig, frame):
    global stop_crawling
    print("Received CTRL+C. Stopping crawling...")
    stop_crawling = True


def is_revenue_domain(url):
    """
    Check if the given URL is within the revenue.ie domain.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return domain.endswith('revenue.ie')

def normalize_url(url, base_url):
    """
    Normalize the URL to handle relative paths and ensure it is complete.
    """
    return urljoin(base_url, url)

# Function to crawl a URL
def crawl_url(url):
    if stop_crawling:
        return

    print("crawling: " + str(url))
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("\tStatus 200: " + str(url))
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Process the page as needed (e.g., extract data, follow links, etc.)
            # You can add your scraping logic here
            
            # Add the current URL to the set of visited URLs
            with visited_urls_lock:
                if url not in visited_urls:
                    visited_urls.add(url)
                    with open(visited_urls_file, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([url])
            
            # Create an XML entry for the current URL and add it to the sitemap
            url_element = ET.SubElement(sitemap, "url")
            loc_element = ET.SubElement(url_element, "loc")
            loc_element.text = url
    
            # Find and add new URLs to the queue of URLs to visit
            for link in soup.find_all('a', href=True):
                new_url = link['href']
                # check we haven't visted this URL before
                if new_url not in visited_urls and is_revenue_domain(new_url):
                    url_queue.put(new_url)
                    with open(url_queue_file, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([new_url])
                    print("\t\tAdded URL to queue: " + str(new_url))
    
    except Exception as e:
        print(f"Error: {e}")

# Function to start crawling
def start_crawling():
    while not stop_crawling:
        # Get the next URL from the queue
        try:
            url = url_queue.get(timeout=1)  # Timeout to periodically check if stop_crawling flag is set
        except Empty:
            continue

        # Crawl the URL
        crawl_url(url)
        
        # Mark the URL as done in the queue
        url_queue.task_done()

# Set up signal handler for CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# Create and start crawler threads
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=start_crawling)
    thread.daemon = True
    thread.start()
    threads.append(thread)

# Add the starting URL to the queue
url_queue.put(start_url)

# Wait for all threads to finish
url_queue.join()

# Join all threads before exiting
for thread in threads:
    thread.join()

# Create a nicely formatted XML string from the sitemap
xmlstr = minidom.parseString(ET.tostring(sitemap)).toprettyxml(indent="  ")

print("Creating sitemap.xml")
# Save the sitemap to a file
with open('tax_sitemap1.xml', 'w') as f:
    f.write(xmlstr)

# Crawling is complete when there are no more URLs to visit
print("Crawling complete, sitemap saved to sitemap.xml!")
