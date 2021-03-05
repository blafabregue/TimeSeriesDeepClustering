"""
Script to compute KNN graphs for SDCN method

Author:
Baptiste Lafabregue 2019.25.04
"""
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='create KNN graphs for UCR or MTS repository data sets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--archives', type=str, metavar='D', required=True,
                        help='archive name')
    args = parser.parse_args()
    root_dir = '.'
    dataset_name = args.dataset
    archive_name = args.archives

    train_dict = utils.read_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]
    graph_path = root_dir + '/graphs/'
    utils.create_directory(graph_path)
    utils.construct_knn_graph(x_train, y_train, graph_path + dataset_name + '.npy', n_jobs=32)
