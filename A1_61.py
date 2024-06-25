import logging
import time
from pyrosm import get_data, OSM
from memory_profiler import profile, LogFile
import networkx as nx
import geopy.distance
import folium
import string
import psutil
import prettytable as pt
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import csv
import math
import numpy as np
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

def rotating_calipers(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    n = len(hull_points)

    if n < 2:
        return 0, None

    max_distance = 0
    max_pair = (None, None)

    j = 1
    for i in range(n):
        next_i = (i + 1) % n
        while True:
            next_j = (j + 1) % n
            d = np.linalg.norm(hull_points[next_i] - hull_points[j])
            if d > max_distance:
                max_distance = d
                max_pair = (hull_points[next_i], hull_points[j])
            
            if np.cross(hull_points[next_i] - hull_points[i], hull_points[next_j] - hull_points[j]) < 0:
                break
            j = next_j

    return max_distance, max_pair

def mercator_projection(lat, lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = lon_rad
    y = math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y

def inverse_mercator_projection(x, y):
    lon = math.degrees(x)
    lat = math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
    return lat, lon

projected_coords = []

with open('crime.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        if lat == 0 and lon == 0:
            pass
        x, y = mercator_projection(lat, lon)
        projected_coords.append((x, y))

projected_coords_array = np.array(projected_coords)

print(projected_coords_array)
max_dist, max_dist_pair = rotating_calipers(projected_coords_array)
point_1 = inverse_mercator_projection(max_dist_pair[0][0], max_dist_pair[0][1])
point_2 = inverse_mercator_projection(max_dist_pair[1][0], max_dist_pair[1][1])
actual_max_dist = geopy.distance.geodesic(point_1, point_2).meters
print(actual_max_dist)

tree = KDTree(projected_coords_array)

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('pathfinding.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def setup_memory_profiler():
    memory_logfile = LogFile('memory_profiler.log', reportIncrementFlag=False)
    return memory_logfile

def osm_get_data(location, logger):
    logger.info(f"Fetching OSM data for {location}")
    return OSM(get_data(location))

def heuristic(u, v, G, visited_nodes):
    visited_nodes.add(u)
    # print(visited_nodes)
    c1 = (G.nodes[u]['y'], G.nodes[u]['x'])
    c2 = (G.nodes[v]['y'], G.nodes[v]['x'])
    radius = 0.00001
    x1, y1 = mercator_projection(G.nodes[u]['y'], G.nodes[u]['x'])
    a1 = np.array([x1, y1]) 
    x2, y2 = mercator_projection(G.nodes[v]['y'], G.nodes[v]['x'])
    a2 = np.array([x2, y2]) 
    get1 = tree.query_ball_point(a1, radius)
    get2 = tree.query_ball_point(a2, radius)
    return geopy.distance.geodesic(c1, c2).meters * 10 + (len(get1) + len(get2)) * 90

@profile(stream=setup_memory_profiler())
def get_greedy_path(start_node, end_node, G, logger):
    logger.info(f"Starting Greedy Best-First Search from node {start_node} to node {end_node}")
    start_time = time.time()
    
    visited_nodes = set()
    path = nx.astar_path(G, start_node, end_node, heuristic=lambda u, v: heuristic(u, v, G, visited_nodes), weight=0)
    end_time = time.time()
    logger.info(f"Greedy Best-First Search Execution Time: {end_time - start_time} seconds")
    return path, end_time - start_time, visited_nodes

@profile(stream=setup_memory_profiler())
def get_astar_path(start_node, end_node, G, logger):
    logger.info(f"Starting A* Search from node {start_node} to node {end_node}")
    start_time = time.time()

    visited_nodes = set()
    path = nx.astar_path(G, start_node, end_node, heuristic=lambda u, v: heuristic(u, v, G, visited_nodes))
    end_time = time.time()
    logger.info(f"A* Search Execution Time: {end_time - start_time} seconds")
    return path, end_time - start_time, visited_nodes

@profile(stream=setup_memory_profiler())
def get_weighted_astar_path(start_node, end_node, G, logger, weight=1.0):
    logger.info(f"Starting Weighted A* Search from node {start_node} to node {end_node} with weight={weight}")
    start_time = time.time()
    
    visited_nodes = set()
    path = nx.astar_path(G, start_node, end_node, heuristic=lambda u, v: weight * heuristic(u, v, G, visited_nodes))
    end_time = time.time()
    logger.info(f"Weighted A* Search Execution Time: {end_time - start_time} seconds")
    return path, end_time - start_time, visited_nodes

def draw_map(start_node, end_node, G, path, visited_nodes, algorithm_name, logger):
    logger.info(f"Drawing map with path from node {start_node} to node {end_node} using {algorithm_name}")
    map_center = [G.nodes[start_node]['y'], G.nodes[start_node]['x']]
    m = folium.Map(location=map_center, zoom_start=13)

    path_latlng = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
    folium.Marker(location=path_latlng[0], icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=path_latlng[-1], icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine(locations=path_latlng, color='blue').add_to(m)

    heat_data = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in visited_nodes]
    HeatMap(heat_data, radius=5, blur=10, max_zoom=1).add_to(m)

    valid_chars = set("-_.() %s%s" % (string.ascii_letters, string.digits))
    map_file = ''.join(c if c in valid_chars else '_' for c in algorithm_name.lower()) + '.html'
    m.save(map_file)
    logger.info(f"Map saved to {map_file}")

def display_comparison_table_and_plots(greedy_time, astar_time, weighted_astar_time):
    # Create a PrettyTable for textual display
    table = pt.PrettyTable()
    table.field_names = ["Algorithm", "Time (seconds)", "Memory (MB)"]
    
    greedy_memory = psutil.Process().memory_info().rss / 1024 / 1024
    astar_memory = psutil.Process().memory_info().rss / 1024 / 1024
    weighted_astar_memory = psutil.Process().memory_info().rss / 1024 / 1024

    table.add_row(["Greedy Best-First Search", greedy_time, greedy_memory])
    table.add_row(["A* Search", astar_time, astar_memory])
    table.add_row(["Weighted A* Search", weighted_astar_time, weighted_astar_memory])

    print(table)

    data = {
        "Algorithm": ["Greedy Best-First Search", "A* Search", "Weighted A* Search"],
        "Time (seconds)": [greedy_time, astar_time, weighted_astar_time],
        "Memory (MB)": [greedy_memory, astar_memory, weighted_astar_memory]
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    plt.bar(df["Algorithm"], df["Time (seconds)"], color=['blue', 'green', 'red'])
    plt.title("Algorithm Time Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(df["Algorithm"], df["Memory (MB)"], color=['blue', 'green', 'red'])
    plt.title("Algorithm Memory Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45)
    plt.show()

def main():
    logger = setup_logging()

    location = "vancouver"
    osm = osm_get_data(location, logger)
    drive_net_nodes, drive_net_edges = osm.get_network(network_type="driving", nodes=True)
    G = osm.to_graph(nodes=drive_net_nodes, edges=drive_net_edges, graph_type="networkx")

    start_node, end_node = list(G.nodes)[372894], list(G.nodes)[404971]
    logger.info(f"Start node: {G.nodes[start_node]}, End node: {G.nodes[end_node]}")

    greedy_path, greedy_time, greedy_nodes = get_greedy_path(start_node, end_node, G, logger)
    astar_path, astar_time, normal_nodes = get_astar_path(start_node, end_node, G, logger)
    weighted_astar_path, weighted_astar_time, weighted_nodes = get_weighted_astar_path(start_node, end_node, G, logger)

    draw_map(start_node, end_node, G, greedy_path, greedy_nodes, "Greedy Best-First Search", logger)
    draw_map(start_node, end_node, G, astar_path, normal_nodes, "A* Search", logger)
    draw_map(start_node, end_node, G, weighted_astar_path, weighted_nodes, "Weighted A* Search", logger)

    display_comparison_table_and_plots(greedy_time, astar_time, weighted_astar_time)

if __name__ == "__main__":
    main()
    
    import requests
from bs4 import BeautifulSoup as bs
from collections import deque
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download('punkt')

SEARCH_DEPTH = 3
PER_LEVEL_LIMIT = 2

def write_to_file(outputfile, text):
    with open(outputfile, 'a', encoding='utf-8') as file:
        file.write(text)
        file.write('\n')

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

from transformers import RobertaTokenizer, RobertaModel, T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import jaccard_similarity_score
from nltk.tokenize import sent_tokenize

def find_relevant_sections(text, prompt, threshold=0.3):  # Lower threshold for Jaccard
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    sentences = sent_tokenize(text)
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    prompt_embedding = model(**prompt_tokens).pooler_output

    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    prompt_vector = vectorizer.transform([prompt])

    relevant_sections = []
    for i, sentence in enumerate(sentences):
        similarity = jaccard_similarity_score(sentence_vectors[i], prompt_vector)
        if similarity >= threshold:
            relevant_sections.append(sentence)

    return relevant_sections

def summarize_text(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    input_ids = tokenizer("summarize: " + text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def filter(links):
    # Create a set of unwanted domains for fast lookup
    unwanted_domains = {"instagram.com", "youtube.com", "google.com", "facebook.com", "twitter.com"}

    # Use a generator expression for concise and efficient filtering
    filtered_links = [
        lnk 
        for lnk in links 
        if "//" in lnk 
        and lnk.split('//')[-1].split('/')[0] not in unwanted_domains
        and not (lnk.split('//')[-1].split('/')[0].startswith("wikipedia.") 
                 and lnk.split('//')[-1].split('/')[0].split('.')[0] != "en")
    ]

    return filtered_links

import requests
from bs4 import BeautifulSoup as bs
import re

def fetch_google_links(query):
    links = set()  # Use a set to store links for automatic deduplication
    url = f"https://www.google.com/search?q={query}&hl=en"

    headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = bs(response.text, "lxml")
        link_pattern = re.compile(r"^https?://")  # Match links starting with http:// or https://
        
        for a_tag in soup.find_all("a", href=link_pattern):  # Directly filter 'a' tags with href
            links.add(a_tag["href"])

    return list(links)  # Convert set back to list if needed

def dfs_visit(url, depth, prompt):
    if depth == SEARCH_DEPTH:
        return []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = bs(response.content, 'html.parser')
        write_to_file("dfs.txt", soup.get_text())

        print("what")
        links = [lnk['href'] for lnk in soup.find_all('a', href=True) if lnk['href'].startswith("http")]
        links = filter(links)

        extracted_links = []
        for link in links[:PER_LEVEL_LIMIT]:
            extracted_links.extend(dfs_visit(link, depth + 1, prompt))
        return links[:PER_LEVEL_LIMIT] + extracted_links
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def bfs_visit(starting_links, depth):
    visited = set()
    queue = deque([(link, 0) for link in starting_links[:PER_LEVEL_LIMIT]])
    result_links = []

    while queue:
        url, level = queue.popleft()
        if level == depth or url in visited:
            continue

        visited.add(url)
        try:
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(url, headers=headers)
            soup = bs(response.content, 'html.parser')
            write_to_file("bfs.txt", soup.get_text())
            links = [lnk['href'] for lnk in soup.find_all('a', href=True) if lnk['href'].startswith("http")]
            links = filter(links)

            result_links.extend(links[:PER_LEVEL_LIMIT])
            for link in links[:PER_LEVEL_LIMIT]:
                queue.append((link, level + 1))
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return result_links

def util_dfs(links, prompt):
    res = []
    print(links[:PER_LEVEL_LIMIT])
    for lnk in links[:PER_LEVEL_LIMIT]:
        tmp_res = dfs_visit(lnk, 0, prompt)
        res.extend(tmp_res)
    return res

def main(query, prompt):
    google_links = filter(fetch_google_links(query=query))
    # google_links = ['https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://en.wikipedia.org/wiki/Caroline_Celico', 'https://en.wikipedia.org/wiki/Marcelo_Saragosa', 'https://en.wikipedia.org/wiki/Gama,_Federal_District', 'https://en.wikipedia.org/wiki/Eduardo_Delani', 'https://en.wikipedia.org/wiki/Kak%C3%A1', 'https://www.transfermarkt.com/kaka/profil/spieler/3366', 'https://www.transfermarkt.com/kaka/profil/spieler/3366', 'https://www.soccerconference.org/personnel/ricardo-izecson-kaka/', 'https://www.soccerconference.org/personnel/ricardo-izecson-kaka/', 'https://www.gettyimages.com/photos/ricardo-izecson-dos-santos-leite', 'https://www.gettyimages.com/photos/ricardo-izecson-dos-santos-leite', 'https://sports.ndtv.com/football/players/90532-kak%C3%A1-playerprofile', 'https://sports.ndtv.com/football/players/90532-kak%C3%A1-playerprofile', 'https://commons.wikimedia.org/wiki/File:Ricardo_Izecson_dos_Santos_Leite_(Kak%C3%A1)_01.jpg', 'https://commons.wikimedia.org/wiki/File:Ricardo_Izecson_dos_Santos_Leite_(Kak%C3%A1)_01.jpg', 'https://www.shutterstock.com/search/ricardo-izecson-dos-santos-leite', 'https://www.shutterstock.com/search/ricardo-izecson-dos-santos-leite', 'https://www.researchgate.net/figure/Figure-2-Ricardo-Izecson-dos-Santos-Leite-was-introduced-in-2009-by-Real-Madrid-as_fig2_331481610', 'https://www.researchgate.net/figure/Figure-2-Ricardo-Izecson-dos-Santos-Leite-was-introduced-in-2009-by-Real-Madrid-as_fig2_331481610']
    print("Google links:", google_links)

    print('DFS:')
    dfs_links = util_dfs(google_links, prompt)
    print(len(dfs_links), dfs_links)

    print('BFS:')
    bfs_links = bfs_visit(google_links, SEARCH_DEPTH)
    print(len(bfs_links), bfs_links)
    
    text = read_text_from_file("dfs.txt")
    relevant_sections = find_relevant_sections(text, prompt)

    summaries = [summarize_text(section) for section in relevant_sections]
    final_summary = '\n'.join(summaries)

    print(final_summary)
    write_to_file("summary_dfs.txt", final_summary)
    
    text = read_text_from_file("bfs.txt")
    relevant_sections = find_relevant_sections(text, prompt)

    print(final_summary)
    summaries = [summarize_text(section) for section in relevant_sections]
    final_summary = '\n'.join(summaries)
    
    write_to_file("summary_bfs.txt", final_summary)

main("Stephen Hawking", "Stephen Hawking and Black Holes")