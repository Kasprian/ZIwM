columns = [
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses',
        'Class'
    ]

params = [{'hidden_layer_sizes': (8,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500},
          {'hidden_layer_sizes': (8,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False, 'max_iter': 500},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False, 'max_iter': 500},
          {'hidden_layer_sizes': (23,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500},
          {'hidden_layer_sizes': (23,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False, 'max_iter': 500}]