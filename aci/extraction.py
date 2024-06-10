#!/usr/bin/env python3

import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL of the page containing the table
url = "https://psmsl.org/data/obtaining/index.php"

# Send a request to fetch the HTML content
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Locate the table
table = soup.find('table')  # Assuming there's only one table

# Extract the table headers
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())

# Extract the table rows
rows = []
for tr in table.find_all('tr')[1:]:  # Skip the header row
    cells = []
    for td in tr.find_all('td'):
        cells.append(td.text.strip())
    rows.append(cells)

# Create a DataFrame and save it as a CSV
df = pd.DataFrame(rows, columns=headers)
df.to_csv('psmsl_data.csv', index=False)

print("Data has been saved to psmsl_data.csv")

