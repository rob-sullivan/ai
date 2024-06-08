import requests
from bs4 import BeautifulSoup
import threading
from queue import Queue
import xml.etree.ElementTree as ET
from xml.dom import minidom
import signal
import sys

# Define the starting URL
start_url = 'https://revenue.ie/en/home.aspx'

# Initialize a set to keep track of visited URLs
visited_urls = set()

# Initialize a queue for URLs to visit
url_queue = Queue()

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
                visited_urls.add(url)
            
            # Create an XML entry for the current URL and add it to the sitemap
            url_element = ET.SubElement(sitemap, "url")
            loc_element = ET.SubElement(url_element, "loc")
            loc_element.text = url
    
            # Find and add new URLs to the queue of URLs to visit
            for link in soup.find_all('a', href=True):
                new_url = link['href']
                # check we haven't visted this URL before
                if new_url not in visited_urls:
                    url_queue.put(new_url)
                    print("\t\tAdded URL to queue: " + str(new_url))
    
    except Exception as e:
        print(f"Error: {e}")

# Function to start crawling
def start_crawling():
    while not stop_crawling:
        # Get the next URL from the queue
        try:
            url = url_queue.get(timeout=1)  # Timeout to periodically check if stop_crawling flag is set
        except queue.Empty:
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
