import os
def parse_graph(file_content):
    """ Parses the graph information from the file content in DIMACS format. """
    edges = []
    for line in file_content:
        if line.startswith('e'):
            _, v1, v2 = line.split()
            edges.append((int(v1), int(v2)))
    return edges

def generate_cnf(graph_edges, num_vertices, num_colors):
    """ Generates the CNF clauses for the graph coloring problem. """
    clauses = []

    # At least one color per vertex
    for vertex in range(1, num_vertices + 1):
        clauses.append([vertex + i * num_vertices for i in range(num_colors)])

    # No more than one color per vertex
    for vertex in range(1, num_vertices + 1):
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                clauses.append([-(vertex + c1 * num_vertices), -(vertex + c2 * num_vertices)])

    # Adjacent vertices must have different colors
    for (v1, v2) in graph_edges:
        for color in range(num_colors):
            clauses.append([-(v1 + color * num_vertices), -(v2 + color * num_vertices)])

    return clauses

def cnf_to_dimacs(cnf_clauses, num_variables):
    """ Converts CNF clauses into DIMACS format. """
    dimacs = f"p cnf {num_variables} {len(cnf_clauses)}\n"
    for clause in cnf_clauses:
        dimacs += " ".join(map(str, clause)) + " 0\n"
    return dimacs



# 着色规则：
# 1. 定义变量：首先，为图中的每个顶点和每种可能的颜色定义一个布尔变量。如果图有n个顶点且需要k种颜色，则会有n * k个变量。例如，如果顶点i可以被染成颜色j，则相应的变量表示为v[i,j]。
# 2. 至少一种颜色：对于每个顶点，至少有一种颜色可以被应用。这可以通过为每个顶点添加一个子句来表示，该子句包含该顶点所有可能颜色的变量的析取。例如，对于顶点i，子句是(v[i,1] OR v[i,2] OR ... OR v[i,k])。
# 3. 最多一种颜色：每个顶点不能被染成超过一种颜色。这意味着对于每个顶点的每对颜色j和l（j != l），必须添加子句(NOT v[i,j] OR NOT v[i,l])。
# 4. 邻接顶点不同颜色：如果两个顶点是相邻的（即它们之间有边相连），它们不能被染成相同的颜色。对于每对相邻顶点i和m，以及每种颜色j，添加子句(NOT v[i,j] OR NOT v[m,j])。
# 5. 生成CNF格式：将以上步骤中生成的所有子句结合起来，形成一个大的CNF公式。每个子句都是变量的析取，整个公式是所有这些子句的合取

if __name__ == '__main__':
    # Reading the contents of the uploaded file to determine its format
    folder_path = './instances/'
    # Output file path
    output_file_path_template = './cnf/{}_color{}.cnf'

    col_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.col')]

    # 读 txt
    record_graphInfo = {} # fname: (vertex num , color_num)
    with open('descrip_from_website.txt', 'r' , encoding='utf-8') as f: # Use this to get best color num
        for l in f.readlines():
            row = l.strip().split(',')
            fname = row[0].split()[0]
            vertices_num = int( row[0].split()[-1].replace( '(' , '' ).strip() )
            try:
                optimal_color = int( row[-2].strip() )
                record_graphInfo[fname] = (vertices_num , optimal_color)
            except:
                print('no optimal color...')
                print(fname)

    for file_name in col_files:
        if file_name not in record_graphInfo: continue
        file_path = os.path.join(folder_path, file_name)
        # start transform
        with open(file_path, 'r') as file:
            content = file.readlines()
        # Displaying the first few lines of the file for format inspection
        print( content[:5] )

        # Parse the graph
        edges = parse_graph(content)
        # Number of vertices (from the file) and assuming number of colors (this can be changed)

        num_vertices , num_colors  = record_graphInfo[file_name]
        for color in [num_colors-1 , num_colors+1, num_colors ]:
            # Generate CNF
            cnf_clauses = generate_cnf(edges, num_vertices, color)
            cnf_content = cnf_to_dimacs(cnf_clauses, num_vertices * color)
            # Writing the CNF to a new file
            with open(output_file_path_template.format(file_name , color), 'w' ,encoding='utf-8') as file:
                file.write(cnf_content)