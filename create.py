import pandas as pd
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import random
from helpers.df_helpers import documents2Dataframe
from helpers.df_helpers import df2Graph
from helpers.df_helpers import graph2Df
import networkx as nx
import seaborn as sns
import csv

data_dir = "cureus"
inputdirectory = Path(f"./data_input/{data_dir}")
out_dir = data_dir
outputdirectory = Path(f"./data_output/{out_dir}")
txt_files = list(inputdirectory.glob("*.txt"))


def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


for txt_file in txt_files:

    with open(txt_file, encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # 创建自定义的 Document 对象
    document = Document(
        page_content=text,
        metadata={"source": txt_file}
    )

    # 处理 Document 对象
    documents = [document]
    # documents = single_file_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    pages = splitter.split_documents(documents)

    print(f"Number of chunks for {txt_file.name} = ", len(pages))

    df = documents2Dataframe(pages)
    # print(f"Shape of DataFrame for {txt_file.name} = ", df.shape)
    concepts_list = df2Graph(df, model='gpt-3.5-turbo')

    dfg1 = graph2Df(concepts_list)
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    base_filename = txt_file.stem
    df.to_csv(outputdirectory / f"{base_filename}_chunks.csv", sep="|", index=False, encoding="utf-8",
              quoting=csv.QUOTE_NONE, escapechar='|')
    dfg1.to_csv(outputdirectory / f"{base_filename}_graph.csv", sep="|", index=False, encoding="utf-8",
                quoting=csv.QUOTE_NONE, escapechar='|')

    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4
    ## Increasing the weight of the relation to 4.
    ## We will assign the weight of 1 when later the contextual proximity will be calculated.
    print(dfg1.shape)

    dfg2 = contextual_proximity(dfg1)
    dfg2.tail()

    dfg = pd.concat([dfg1, dfg2], axis=0)
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()

    G = nx.Graph()

    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )

    ## Add edges to the graph
    for index, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count'] / 4
        )

    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print("Number of Communities = ", len(communities))
    print(communities)

    palette = "hls"

    colors = colors2Community(communities)

    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]

from pyvis.network import Network

graph_output_directory = f"./docs/{base_filename}_index.html"

net = Network(
    notebook=False,
    # bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    # font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
net.show_buttons(filter_=["physics"])
html_content = net.generate_html()

# Write the HTML content to a file with UTF-8 encoding
with open(graph_output_directory, 'w', encoding='utf-8') as f:
    f.write(html_content)
# net.show(graph_output_directory, notebook=False)

# 保存节点信息
nodes_data = []
for node in G.nodes(data=True):
    nodes_data.append({
        'node': node[0],
        'group': node[1].get('group'),
        'color': node[1].get('color'),
        'size': node[1].get('size')
    })

df_nodes = pd.DataFrame(nodes_data)
df_nodes.to_csv(f"./nodes/{base_filename}_nodes_data.csv", index=False)

# 保存边信息
edges_data = []
for edge in G.edges(data=True):
    edges_data.append({
        'node_1': edge[0],
        'node_2': edge[1],
        'weight': edge[2].get('weight')
    })

df_edges = pd.DataFrame(edges_data)
df_edges.to_csv(f"./nodes/{base_filename}_edges_data.csv", index=False)
communities_data = []
for group, community in enumerate(communities, 1):
    for node in community:
        communities_data.append({'node': node, 'group': group})

df_communities = pd.DataFrame(communities_data)
df_communities.to_csv(f"./nodes/{base_filename}_communities_data.csv", index=False)
