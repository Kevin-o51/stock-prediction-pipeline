import requests
from bs4 import BeautifulSoup

def fetch_sp500_tickers():
    """
    Scrapes Wikipedia for the current S&P 500 tickers
    and returns them as a list of symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    response.raise_for_status()  # raise an error for bad status
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", {"id": "constituents"})
    symbols = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if cols:
            symbol = cols[0].text.strip()
            # Some tickers have periods or other chars, we can clean them up if needed
            symbol = symbol.replace(".", "-")
            symbols.append(symbol)
    return symbols

if __name__ == "__main__":
    # Fetch and save the tickers to CSV
    tickers = fetch_sp500_tickers()
    with open("sp500_tickers.csv", "w") as f:
        for t in tickers:
            f.write(t + "\n")
    print(f"Successfully saved {len(tickers)} tickers to sp500_tickers.csv")

