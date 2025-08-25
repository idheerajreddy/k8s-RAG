import requests
from bs4 import BeautifulSoup
import os

# List of URLs to scrape (add more as needed)
URLS = [
    "https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/",
    "https://kubernetes.io/docs/concepts/architecture/",
    "https://blog.kubernetes.io/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/troubleshoot-kubectl/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-usage-monitoring/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/monitor-node-health/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/crictl/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/,"
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/kubectl-node-debug/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/local-debugging/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/windows/",

]

OUTPUT_DIR = "scraped_content"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Get text
    text = soup.get_text(separator="\n")
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def save_content(url, text):
    filename = os.path.join(OUTPUT_DIR, url.replace("https://", "").replace("/", "_") + ".txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {filename}")

def main():
    for url in URLS:
        print(f"Scraping: {url}")
        html = fetch_content(url)
        if html:
            text = parse_html(html)
            save_content(url, text)

if __name__ == "__main__":
    main()
