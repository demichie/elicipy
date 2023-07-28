from ete3 import Tree, TreeStyle, faces, AttrFace, CircleFace
import textwrap
import pandas as pd
import numpy as np
import os


def layout(node):

    # Add node name to laef nodes
    N = AttrFace("name", fsize=9, fgcolor="black")

    # Split short Q in multiple lines
    lines = textwrap.wrap(code2name[node.name], 35, break_long_words=False)
    # Add newline character at the end of lines
    lines = "\n".join(line.strip() for line in lines)

    if node.idx > 0:

        # add faces only when first node is a real node

        # add node name
        faces.add_face_to_node(N, node, 0, position="branch-top")

        # add Cooke percentiles
        cookeFace = faces.TextFace(
            "Cooke [" + str(node.q5) + "," + str(node.q50) + "," +
            str(node.q95) + "]",
            fsize=8,
            fgcolor="RoyalBlue",
        )
        faces.add_face_to_node(cookeFace,
                               node,
                               column=0,
                               position="branch-bottom")

        # add erf percentiles
        erfFace = faces.TextFace(
            "ERF [" + str(node.q_erf5) + "," + str(node.q_erf50) + "," +
            str(node.q_erf95) + "]",
            fsize=8,
            fgcolor="Red",
        )
        faces.add_face_to_node(erfFace,
                               node,
                               column=0,
                               position="branch-bottom")

        # add EW percentiles
        EWFace = faces.TextFace(
            "EW [" + str(node.q_EW5) + "," + str(node.q_EW50) + "," +
            str(node.q_EW95) + "]",
            fsize=8,
            fgcolor="Green",
        )
        faces.add_face_to_node(EWFace,
                               node,
                               column=0,
                               position="branch-bottom")

    # add short Q to all the nodes
    longNameFace = faces.TextFace(lines, fsize=6)
    faces.add_face_to_node(longNameFace, node, column=0, position="branch-top")

    if node.is_leaf() and "weight_cooke" in node.features:

        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        C = CircleFace(radius=node.weight_cooke * 100,
                       color="RoyalBlue",
                       style="sphere")
        # Let's make the sphere transparent
        C.opacity = 0.3
        # And place as a float face over the tree
        # faces.add_face_to_node(C, node, 1, position="branch-right")

    if node.is_leaf() and "weight_erf" in node.features:

        # Creates a sphere face whose size is proportional to node's
        # feature "weight"
        C1 = CircleFace(radius=node.weight_erf * 80,
                        color="Red",
                        style="sphere")
        # Let's make the sphere transparent
        C1.opacity = 0.3
        # And place as a float face over the tree
        # faces.add_face_to_node(C1, node, 2, position="branch-right")


def build_tree(csv_file, first_node, first_node_str):

    # read csf files with idx, short Q, percentiles and tree parents
    df = pd.read_csv(csv_file, header=0, index_col=0)

    if first_node < 0:

        # create "fake" node for negative first nodes and add to dataframe
        new_row = {
            "IDX": [first_node],
            "SHORT_Q": [first_node_str],
            "EW_5": [0],
            "EW_50": [0],
            "EW_95": [0],
            "COOKE_5": [0],
            "COOKE_50": [0],
            "COOKE_95": [0],
            "ERF_5": [0],
            "ERF_50": [0],
            "ERF_95": [0],
            "PARENT": [0],
        }
        df_row = pd.DataFrame(data=new_row)
        df_row = df_row.set_index("IDX")
        df = df.append(df_row)

    idx_list = list(df.index)

    parents = np.array((df["PARENT"].tolist()), dtype=int)
    short_q = df["SHORT_Q"].tolist()

    # build tree from list of parents and first node
    child_string = build_subtree(idx_list, parents, first_node,
                                 "TQ" + str(first_node))
    tree_string = child_string[1] + ";"

    t = Tree(tree_string, format=8)

    # Create a conversion between leaf names and real names
    names = ["TQ" + str(i) for i in idx_list]
    code2name = {names[i]: short_q[i] for i in range(len(names))}

    cooke_5s = np.array((df["COOKE_5"].tolist()), dtype=int)
    cooke_50s = np.array((df["COOKE_50"].tolist()), dtype=int)
    cooke_95s = np.array((df["COOKE_95"].tolist()), dtype=int)

    erf_5s = np.array((df["ERF_5"].tolist()), dtype=int)
    erf_50s = np.array((df["ERF_50"].tolist()), dtype=int)
    erf_95s = np.array((df["ERF_95"].tolist()), dtype=int)

    EW_5s = np.array((df["EW_5"].tolist()), dtype=int)
    EW_50s = np.array((df["EW_50"].tolist()), dtype=int)
    EW_95s = np.array((df["EW_95"].tolist()), dtype=int)

    # Add percentile features in all nodes
    for node in t.traverse():

        idx = int(node.name.replace("TQ", "")) - 1

        node.add_features(idx=idx)

        node.add_features(q5=cooke_5s[idx])
        node.add_features(q50=cooke_50s[idx])
        node.add_features(q95=cooke_95s[idx])

        node.add_features(q_erf5=erf_5s[idx])
        node.add_features(q_erf50=erf_50s[idx])
        node.add_features(q_erf95=erf_95s[idx])

        node.add_features(q_EW5=EW_5s[idx])
        node.add_features(q_EW50=EW_50s[idx])
        node.add_features(q_EW95=EW_95s[idx])

    # Add features in all nodes
    for n in t.traverse():
        n.add_features(weight_cooke=1.0)
        n.add_features(weight_erf=1.0)

        node = n

        if node.is_leaf():
            while node:
                n.weight_cooke = n.weight_cooke * node.q50 / 100.0
                n.weight_erf = n.weight_erf * node.q_erf50 / 100.0
                node = node.up

    # Create an empty TreeStyle
    ts = TreeStyle()

    # Set our custom layout function
    ts.layout_fn = layout
    ts.rotation = 0
    # Draw a tree
    # ts.mode = "c"

    # We will add node names manually
    ts.show_leaf_name = False

    # Show branch data
    # ts.show_branch_length = True
    # ts.show_branch_support = True

    ts.show_scale = False
    ts.branch_vertical_margin = 15

    return t, ts, code2name


def build_subtree(idx_list, parents, node, tree_string):

    childs = [idx_list[i] for i, p in enumerate(parents) if p == node]

    rep_string = "TQ" + str(node)
    child_string = "(TQ" + ",TQ".join(map(str, childs)) + ")"
    if len(childs) > 0:
        tree_string = tree_string.replace(rep_string,
                                          child_string + rep_string)

    for ch in childs:

        childs, tree_string = build_subtree(idx_list, parents, ch, tree_string)

    return childs, tree_string


if __name__ == "__main__":

    from ElicipyDict import output_dir
    from ElicipyDict import Repository
    from ElicipyDict import first_node_list
    from ElicipyDict import first_node_str_list
    from ElicipyDict import elicitation_name

    # get current path
    path = os.getcwd()
    # change current path to elicitation folder
    path = path + "/" + Repository
    print("Path", path)

    os.chdir(path)

    output_dir = path + "/" + output_dir

    csv_file = "tree.csv"

    for first_node, first_node_str in zip(first_node_list,
                                          first_node_str_list):

        t, ts, code2name = build_tree(csv_file, first_node, first_node_str)

        t.render(
            output_dir + "/" + elicitation_name + "_tree" + str(first_node) +
            ".png",
            units="in",
            w=3,
            dpi=600,
            tree_style=ts,
        )

    import matplotlib.pyplot as plt

    v1 = [2, 5, 3, 1, 4]
    labels1 = ["A", "B", "C", "D", "E"]
    v2 = [4, 1, 3, 4, 1]
    labels2 = ["V", "W", "X", "Y", "Z"]
    width = 0.3
    wedge_properties = {"width": width, "edgecolor": "w", "linewidth": 2}

    plt.pie(v1,
            labels=labels1,
            labeldistance=0.85,
            wedgeprops=wedge_properties)
    plt.pie(
        v2,
        labels=labels2,
        labeldistance=0.75,
        radius=1 - width,
        wedgeprops=wedge_properties,
    )
    plt.show()

    # t.show(tree_style=ts)
