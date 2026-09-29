import json
import plotly.graph_objects as go
import os

def generate_3d_visualization(input_json_path, output_html_path):
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    nodes = data['nodes']
    edges = data.get('edges', [])
    
    # 1. Node Data Preparation
    # Group by chain
    chain_groups = {}

    node_dict = {}
    
    for i, node in enumerate(nodes):
        chain = node.get('chain', 'A')
        if chain not in chain_groups:
            chain_groups[chain] = []
            
        coords = node.get('coords', [0, 0, 0])
        
        phi = node.get('phi')
        psi = node.get('psi')
        phi_str = f"{phi:.2f}" if phi is not None else "N/A"
        psi_str = f"{psi:.2f}" if psi is not None else "N/A"
        
        hover_text = (
            f"<b>{node.get('resname')} {node.get('residue_number')}</b> (Chain {chain})<br>"
            f"XYZ: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})<br>"
            f"Charge: {node.get('charge', 0)}<br>"
            f"Hydropathy: {node.get('hydropathy', 0)}<br>"
            f"Volume: {node.get('volume', 0)}<br>"
            f"H-Donors: {node.get('h_donors', 0)}<br>"
            f"H-Acceptors: {node.get('h_acceptors', 0)}<br>"
            f"Phi: {phi_str}<br>"
            f"Psi: {psi_str}<br>"
            f"B-factor: {node.get('b_factor', 'N/A')}"
        )
        
        node_info = {
            'idx': i,
            'x': coords[0], 'y': coords[1], 'z': coords[2],
            'size': 10,
            'text': hover_text,
            'label': f"{node.get('resname')} {node.get('residue_number')}"
        }
        chain_groups[chain].append(node_info)
        node_dict[i] = node_info
        
    fig = go.Figure()
    
    # Colors for Chains
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    chain_colors = {chain: colors[i % len(colors)] for i, chain in enumerate(chain_groups.keys())}
    
    # 2. Add Node Traces
    node_traces_count = 0
    for chain, items in chain_groups.items():
        if not items: continue
        fig.add_trace(go.Scatter3d(
            x=[item['x'] for item in items],
            y=[item['y'] for item in items],
            z=[item['z'] for item in items],
            mode='markers', # Toggle labels via updatemenus
            text=[item['label'] for item in items],
            hovertext=[item['text'] for item in items],
            hoverinfo='text',
            name=f"Chain {chain}",
            marker=dict(
                size=[item['size'] for item in items],
                color=chain_colors[chain],
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.9,
                sizemode='diameter'
            ),
            textposition="top center",
            textfont=dict(size=10, color='black'),
            legendgroup="nodes",
            legendgrouptitle_text="Chains" if node_traces_count == 0 else None
        ))
        node_traces_count += 1

    # 3. Add Spatial Connections
    # Group edges by distance to vary thickness and opacity
    edge_bins = {
        'Close Contact (< 5Å)': {'x': [], 'y': [], 'z': [], 'width': 3, 'color': 'rgba(0, 0, 0, 0.8)'},
        'Medium Spatial (5-8Å)': {'x': [], 'y': [], 'z': [], 'width': 1.5, 'color': 'rgba(0, 0, 0, 0.4)'},
        'Weak Spatial (> 8Å)': {'x': [], 'y': [], 'z': [], 'width': 0.5, 'color': 'rgba(0, 0, 0, 0.15)'}
    }
    
    mid_x, mid_y, mid_z, mid_text = [], [], [], []
    
    for edge in edges:
        s_idx, t_idx = edge['source'], edge['target']
        if s_idx >= len(nodes) or t_idx >= len(nodes): continue
        
        s_node, t_node = nodes[s_idx], nodes[t_idx]
        
        x0, y0, z0 = node_dict[s_idx]['x'], node_dict[s_idx]['y'], node_dict[s_idx]['z']
        x1, y1, z1 = node_dict[t_idx]['x'], node_dict[t_idx]['y'], node_dict[t_idx]['z']
        
        dist = edge.get('distance', 0)
        
        if dist < 5.0:
            b = edge_bins['Close Contact (< 5Å)']
        elif dist < 8.0:
            b = edge_bins['Medium Spatial (5-8Å)']
        else:
            b = edge_bins['Weak Spatial (> 8Å)']
            
        b['x'].extend([x0, x1, None])
        b['y'].extend([y0, y1, None])
        b['z'].extend([z0, z1, None])
        
        # Midpoint for hover
        mid_x.append((x0+x1)/2)
        mid_y.append((y0+y1)/2)
        mid_z.append((z0+z1)/2)
        
        is_same = edge.get('is_same_chain', s_node.get('chain') == t_node.get('chain'))
        mid_text.append(
            f"Edge: {s_node.get('resname')} {s_node.get('residue_number')} ➔ "
            f"{t_node.get('resname')} {t_node.get('residue_number')}<br>"
            f"Distance: {dist:.2f} Å<br>"
            f"Same Chain: {is_same}<br>"
            f"Type: {edge.get('edge_type', 'Spatial')}"
        )

    edge_traces_count = 0
    for name, b in edge_bins.items():
        if not b['x']: continue
        fig.add_trace(go.Scatter3d(
            x=b['x'], y=b['y'], z=b['z'],
            mode='lines',
            line=dict(color=b['color'], width=b['width']),
            hoverinfo='none',
            name=name,
            legendgroup="edges",
            legendgrouptitle_text="Spatial Edges" if edge_traces_count == 0 else None
        ))
        edge_traces_count += 1
        
    # Edge hover points
    if mid_x:
        fig.add_trace(go.Scatter3d(
            x=mid_x, y=mid_y, z=mid_z,
            mode='markers',
            marker=dict(size=2, color='rgba(0,0,0,0)'),
            hovertext=mid_text,
            hoverinfo='text',
            showlegend=False,
            name="Edge Details",
            legendgroup="edges"
        ))
        
    # 4. Updatemenus (Controls)
    node_trace_indices = list(range(node_traces_count))
    
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"mode": ['markers'] * node_traces_count}, node_trace_indices], 
                    label="Hide Labels",
                    method="restyle"
                ),
                dict(
                    args=[{"mode": ['markers+text'] * node_traces_count}, node_trace_indices],
                    label="Show Labels",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]

    fig.update_layout(
        title='3D Protein Feature Graph',
        updatemenus=updatemenus,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            xaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            yaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            zaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            x=1.05,
            y=0.9,
            traceorder="grouped",
            font=dict(family="sans-serif", size=12, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        )
    )
    
    print(f"Saving visualization to {output_html_path}...")
    fig.write_html(output_html_path)
    print("Done!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(base_dir, 'data', 'output', 'full_protein_graph.json')
    output_file = os.path.join(base_dir, 'data', 'output', 'protein_3d_graph.html')
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    generate_3d_visualization(input_file, output_file)
