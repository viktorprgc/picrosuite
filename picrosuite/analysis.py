def fire(
    img,
    **params
    ):
    """
    Generate graph with fibers
    ----
    Returns:
    img_edt: distance transform of the value channel
    net_graph: the networkx graph generated
    network: the list of FibreNetwork instances
    metrics: dataframe with metrics on it
    """
    img_ = denoise(img)

    img_edt, net_graph = build_network(
        img_, **params
    )

    network = fibre_network_assignment(net_graph)

    return net_graph


        
