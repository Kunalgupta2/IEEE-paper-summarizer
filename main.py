import os
from flask import Flask, request, render_template, redirect, url_for
import requests
import json
from bs4 import BeautifulSoup
import PyPDF2
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

app = Flask(__name__)
app.config["FILE_UPLOADS"] = "pdfs"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form["topic"]
        pdf_path = search_ieee(topic)
        if pdf_path:
            all_text, summaries = extract_text()  # Get both all_text and summaries
            summary = summarize_text(
                all_text, length=10
            )  # Pass all_text to summarize_text
            return render_template(
                "result.html", pdf_url=pdf_path, summary=summary, summaries=summaries
            )

    return render_template("index.html")


def search_ieee(topic):
    url = "https://ieeexplore.ieee.org/rest/search"

    payload = json.dumps(
        {
            "newsearch": True,
            "queryText": topic,
            "highlight": True,
            "returnFacets": ["ALL"],
            "returnType": "SEARCH",
            "matchPubs": True,
        }
    )

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }

    # Create a session and set headers
    session = requests.Session()
    session.headers = headers
    # Load cookies in the session
    session.get("https://ieeexplore.ieee.org/Xplore/home.jsp")
    # Send search request
    response = session.post(url, data=payload)
    data = response.json()

    # Extract PDF links using BeautifulSoup
    pdf_links = []
    if "records" in data:
        for record in data["records"]:
            pdf_links.append("https://ieeexplore.ieee.org" + (record["pdfLink"]))

    # Download PDFs
    pdf_path = os.path.join(os.curdir + "/pdfs")
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)

    downloaded_pdfs = []

    # Loop through the records and download PDFs
    for idx, pdf_link in enumerate(pdf_links, start=1):
        if pdf_link in downloaded_pdfs:
            # Skip download if this PDF has already been downloaded
            print(f"PDF {idx} already downloaded")
            continue

        pdf_response = session.get(pdf_link)
        if pdf_response.status_code == 200:
            pdf_name = pdf_link.split("/")[-1].split("=")[-1]
            soup = BeautifulSoup(pdf_response.content, features="html5lib")
            iframe = soup.find("iframe")

            if iframe is not None:
                iframe_src = iframe.get("src")
                pdf_response = session.get(iframe_src)
                with open(f"{pdf_path}/{pdf_name}.pdf", "wb+") as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"PDF {idx} downloaded as {pdf_name}")

                # Add the downloaded PDF to the list
                downloaded_pdfs.append(pdf_link)
            else:
                print(f"No iframe found for PDF {idx}, skipping download")
        else:
            print(f"Failed to download PDF {idx}")
    print("All PDFs downloaded.")
    return os.path.join(pdf_path, f"{topic}.pdf")


def extract_text():
    # Get a list of PDF files in the "pdfs" folder
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]

    all_text = ""  # Accumulate text from all PDFs
    summaries = []  # Store summaries for each PDF

    for pdf_file in pdf_files:
        pdf_path = os.path.join("pdfs", pdf_file)

        # Open the PDF file
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Extract text from each page
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()

        # Close the PDF file
        # reader.close()

        # Generate a summary for the current PDF
        pdf_summary = summarize_text(pdf_text, length=10)
        summaries.append(pdf_summary)

        # Add the summary to the list
        all_text += pdf_summary + "\n\n\n"  # Add a newline between summaries

        # Remove the processed PDF
        os.remove(pdf_path)

    return all_text, summaries


def summarize_text(text, length):
    stop_words = set(stopwords.words("english") + list(punctuation))
    words = word_tokenize(text.lower())
    sent_tokens = sent_tokenize(text)
    word_tokens = [word for word in words if word not in stop_words]

    word_freq = FreqDist(word_tokens)
    rank = defaultdict(int)
    for i, sent in enumerate(sent_tokens):
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                rank[i] += word_freq[word]

    indices = nlargest(int(length), rank, key=rank.get)
    final_summary = [sent_tokens[j] for j in indices]
    return " ".join(final_summary)


if __name__ == "__main__":
    app.run(debug=True)
