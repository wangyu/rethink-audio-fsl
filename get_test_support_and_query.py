import pickle as pkl

if __name__ == '__main__':
    test_filelist = pkl.load(open('test_filelist.pkl', 'rb'))

    # get monophonic files
    test_label_to_path = dict()
    for idx in range(len(test_filelist['labels'])):
        if len(test_filelist['labels'][idx]) == 1:
            label = test_filelist['labels'][idx][0]
            if label not in test_label_to_path:
                test_label_to_path[label] = []
            test_label_to_path[label].append(test_filelist['data'][idx])

    # sample monophonic support examples
    novel_tags = list(range(74,89))
    test_support_filelist = {'data': [], 'labels': []}

    for label in novel_tags:
        paths = test_label_to_path[label]
        n_support = int(0.2 * len(paths))
        support = paths[:n_support]
        test_support_filelist['data'] += support
        for n in range(n_support):
            test_support_filelist['labels'].append([label])

    # the rest of files are used as queries at evaluation
    support_idx = [test_filelist['data'].index(f) for f in test_support_filelist['data']]

    test_query_filelist = dict()
    test_query_filelist['data'] = [test_filelist['data'][i] for i in range(len(test_filelist['data'])) if i not in support_idx]
    test_query_filelist['labels'] = [test_filelist['labels'][i] for i in range(len(test_filelist['data'])) if i not in support_idx]

    # save filelists
    with open('test_support_filelist.pkl', 'wb') as f:
        pkl.dump(test_support_filelist, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open('test_query_filelist.pkl', 'wb') as f:
        pkl.dump(test_query_filelist, f, protocol=pkl.HIGHEST_PROTOCOL)