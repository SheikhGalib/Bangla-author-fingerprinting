"""
Scrape all 14 chapters of 'মিসির আলির অমিমাংসিত রহস্য' from ebanglalibrary.com
and save the cleaned Bengali text to corpus.txt in Lab 02 folder.
"""
import os
import re
import sys
import urllib.request
import urllib.parse
from html import unescape

# Force UTF-8 stdout
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = r'E:\4-1\NLP Lab\Lab 02'
OUT_TXT = os.path.join(OUT_DIR, 'corpus.txt')
HTML_DIR = os.path.join(OUT_DIR, 'ebangla_html')
os.makedirs(HTML_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Chapter URLs (already extracted from the index page)
CHAPTERS = [
    ('০১. আপনি কি ভূত দেখেছেন',     'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a7-%e0%a6%86%e0%a6%aa%e0%a6%a8%e0%a6%bf-%e0%a6%95%e0%a6%bf-%e0%a6%ad%e0%a7%82%e0%a6%a4-%e0%a6%a6%e0%a7%87%e0%a6%96%e0%a7%87%e0%a6%9b%e0%a7%87%e0%a6%a8/'),
    ('০২. পুলিশের লোকদের বিব্রত',    'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a8-%e0%a6%aa%e0%a7%81%e0%a6%b2%e0%a6%bf%e0%a6%b6%e0%a7%87%e0%a6%b0-%e0%a6%b2%e0%a7%8b%e0%a6%95%e0%a6%a6%e0%a7%87%e0%a6%b0-%e0%a6%ac%e0%a6%bf%e0%a6%ac%e0%a7%8d%e0%a6%b0%e0%a6%a4/'),
    ('০৩. বিশিষ্ট বেহালাবাদক শিল্পপতি ওসমান গনি', 'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a9-%e0%a6%ac%e0%a6%bf%e0%a6%b6%e0%a6%bf%e0%a6%b7%e0%a7%8d%e0%a6%9f-%e0%a6%ac%e0%a7%87%e0%a6%b9%e0%a6%be%e0%a6%b2%e0%a6%be%e0%a6%ac%e0%a6%be%e0%a6%a6%e0%a6%95-%e0%a6%b6%e0%a6%bf/'),
    ('০৪. অম্বিকাবাবুর বাড়ি',         'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%aa-%e0%a6%85%e0%a6%ae%e0%a7%8d%e0%a6%ac%e0%a6%bf%e0%a6%95%e0%a6%be%e0%a6%ac%e0%a6%be%e0%a6%ac%e0%a7%81%e0%a6%b0-%e0%a6%ac%e0%a6%be%e0%a7%9c%e0%a6%bf/'),
    ('০৫. বৃষ্টিতে ভেজার জন্যে',      'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ab-%e0%a6%ac%e0%a7%83%e0%a6%b7%e0%a7%8d%e0%a6%9f%e0%a6%bf%e0%a6%a4%e0%a7%87-%e0%a6%ad%e0%a7%87%e0%a6%9c%e0%a6%be%e0%a6%b0-%e0%a6%9c%e0%a6%a8%e0%a7%8d%e0%a6%af%e0%a7%87/'),
    ('০৬. হোম মিনিস্টারের দর্শনপ্রার্থী', 'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ac-%e0%a6%b9%e0%a7%8b%e0%a6%ae-%e0%a6%ae%e0%a6%bf%e0%a6%a8%e0%a6%bf%e0%a6%b8%e0%a7%8d%e0%a6%9f%e0%a6%be%e0%a6%b0%e0%a7%87%e0%a6%b0-%e0%a6%a6%e0%a6%b0%e0%a7%8d%e0%a6%b6%e0%a6%a8/'),
    ('০৭. নাদিয়া অবাক হয়ে বললেন',  'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ad-%e0%a6%a8%e0%a6%be%e0%a6%a6%e0%a6%bf%e0%a7%9f%e0%a6%be-%e0%a6%85%e0%a6%ac%e0%a6%be%e0%a6%95-%e0%a6%b9%e0%a7%9f%e0%a7%87-%e0%a6%ac%e0%a6%b2%e0%a6%b2%e0%a7%87%e0%a6%a8/'),
    ('০৮. কড়া নাড়তেই দরজা খুলল',    'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ae-%e0%a6%95%e0%a7%9c%e0%a6%be-%e0%a6%a8%e0%a6%be%e0%a7%9c%e0%a6%a4%e0%a7%87%e0%a6%87-%e0%a6%a6%e0%a6%b0%e0%a6%9c%e0%a6%be-%e0%a6%96%e0%a7%81%e0%a6%b2%e0%a6%b2/'),
    ('০৯. হুইল চেয়ারে যে-বৃদ্ধা বসে আছেন', 'https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%af-%e0%a6%b9%e0%a7%81%e0%a6%87%e0%a6%b2-%e0%a6%9a%e0%a7%87%e0%a7%9f%e0%a6%be%e0%a6%b0%e0%a7%87-%e0%a6%af%e0%a7%87-%e0%a6%ac%e0%a7%83%e0%a6%a6%e0%a7%8d%e0%a6%a7%e0%a6%be-%e0%a6%ac/'),
    ('১০. গুলশান থানার ওসি',          'https://www.ebanglalibrary.com/lessons/%e0%a7%a7%e0%a7%a6-%e0%a6%97%e0%a7%81%e0%a6%b2%e0%a6%b6%e0%a6%be%e0%a6%a8-%e0%a6%a5%e0%a6%be%e0%a6%a8%e0%a6%be%e0%a6%b0-%e0%a6%93%e0%a6%b8%e0%a6%bf/'),
    ('১১. ডাক্তার মুসফেকুর রহমান',     'https://www.ebanglalibrary.com/lessons/%e0%a7%a7%e0%a7%a7-%e0%a6%a1%e0%a6%be%e0%a6%95%e0%a7%8d%e0%a6%a4%e0%a6%be%e0%a6%b0-%e0%a6%ae%e0%a7%81%e0%a6%b8%e0%a6%ab%e0%a7%87%e0%a6%95%e0%a7%81%e0%a6%b0-%e0%a6%b0%e0%a6%b9%e0%a6%ae%e0%a6%be/'),
    ('১২. হোম মিনিস্টার সাহেব',       'https://www.ebanglalibrary.com/lessons/%e0%a7%a7%e0%a7%a8-%e0%a6%b9%e0%a7%8b%e0%a6%ae-%e0%a6%ae%e0%a6%bf%e0%a6%a8%e0%a6%bf%e0%a6%b8%e0%a7%8d%e0%a6%9f%e0%a6%be%e0%a6%b0-%e0%a6%b8%e0%a6%be%e0%a6%b9%e0%a7%87%e0%a6%ac/'),
    ('১৩. মিসির আলি বিছানায় ঘুমুতে গেলেন', 'https://www.ebanglalibrary.com/lessons/%e0%a7%a7%e0%a7%a9-%e0%a6%ae%e0%a6%bf%e0%a6%b8%e0%a6%bf%e0%a6%b0-%e0%a6%86%e0%a6%b2%e0%a6%bf-%e0%a6%ac%e0%a6%bf%e0%a6%9b%e0%a6%be%e0%a6%a8%e0%a6%be%e0%a7%9f-%e0%a6%98%e0%a7%81%e0%a6%ae%e0%a7%81/'),
    ('১৪. নীপবন থাকে শূন্য',          'https://www.ebanglalibrary.com/lessons/%e0%a7%a7%e0%a7%aa-%e0%a6%a8%e0%a7%80%e0%a6%aa%e0%a6%ac%e0%a6%a8-%e0%a6%a5%e0%a6%be%e0%a6%95%e0%a7%87-%e0%a6%b6%e0%a7%82%e0%a6%a8%e0%a7%8d%e0%a6%af/'),
]


def fetch(url, out_path):
    """Download a URL to a file."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(out_path, 'wb') as f:
        f.write(data)
    return len(data)


def extract_text(html):
    """Extract the chapter text from the page HTML."""
    # The content sits inside <div class="entry-content entry-content-single"> ... </div>
    # We use a simple state-machine to grab text within <p> tags inside the article.

    # Find the entry-content block
    m = re.search(r'<div class="entry-content entry-content-single"[^>]*>(.*?)</div>\s*</div>', html, re.DOTALL)
    if not m:
        # fallback: find any <p> blocks
        body = html
    else:
        body = m.group(1)

    # Find each <p>...</p>
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', body, re.DOTALL)

    cleaned = []
    for p in paragraphs:
        # strip inner tags
        text = re.sub(r'<[^>]+>', '', p)
        # decode HTML entities
        text = unescape(text)
        # remove script/style leftovers
        text = re.sub(r'\s+', ' ', text).strip()
        if text and len(text) > 1:
            cleaned.append(text)

    return '\n\n'.join(cleaned)


def main():
    all_text = []
    for i, (title, url) in enumerate(CHAPTERS, 1):
        html_path = os.path.join(HTML_DIR, f'chapter_{i:02d}.html')
        print(f'[{i:02d}/14] Downloading: {title[:30]}...')
        try:
            size = fetch(url, html_path)
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                html = f.read()
            text = extract_text(html)
            print(f'         Got {size//1024} KB HTML, extracted {len(text)} chars')
            if not text:
                print(f'         WARNING: no text extracted; check chapter_{i:02d}.html')
        except Exception as e:
            text = ''
            print(f'         ERROR: {e}')

        # Add to corpus with chapter title
        all_text.append(f'\n\n=== {title} ===\n\n')
        all_text.append(text)

    # Write corpus.txt
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(''.join(all_text))

    total = sum(len(t) for t in all_text)
    print(f'\nDone. Total: {total} chars saved to {OUT_TXT}')


if __name__ == '__main__':
    main()