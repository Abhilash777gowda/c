import requests
from bs4 import BeautifulSoup

url = "https://www.udayavani.com/category/crime"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

try:
    print(f"Fetching {url}...")
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content Length: {len(resp.content)} bytes")
    
    soup = BeautifulSoup(resp.content, "html.parser")
    links = soup.find_all("a")
    print(f"Total links: {len(links)}")
    
    lines = []
    for idx, a in enumerate(links):
        text = a.get_text(strip=True)
        href = a.get("href")
        lines.append(f"Link {idx}: text='{text}', href='{href}'")
        
    with open("scratch/udayavani_links.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Links written to scratch/udayavani_links.txt")
except Exception as e:
    print(f"Error: {e}")
