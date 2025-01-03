timestamp_schema = {
    'type': 'object',
    'properties': {
        'date': {'type': 'string'},
        'time': {'type': 'string'}
    },
    'required': ['date', 'time']
}

proc_prep_parameters_schema = {
    'type': 'object',
    'properties': {
        't_from': timestamp_schema,
        't_to': timestamp_schema,
        'ucar-rda-credentials': {
            'type': 'object',
            'properties': {
                'email': {'type': 'string'},
                'password': {'type': 'string'}
            },
            'required': ['email', 'password']
        }
    },
    'required': ['t_from', 't_to', 'ucar-rda-credentials']
}

proc_sim_parameters_schema = {
    'type': 'object',
    'properties': {
        'pbs_queue': {'type': 'string'},
        'pbs_project_id': {'type': 'string'},
        'pbs_nnodes': {'type': 'number'},
        'pbs_ncpus': {'type': 'number'},
        'pbs_mem': {'type': 'string'},
        'pbs_mpiprocs': {'type': 'number'},
        'walltime': {'type': 'string'}
    },
    'required': ['pbs_queue', 'pbs_project_id', 'pbs_nnodes', 'pbs_ncpus', 'pbs_mem', 'pbs_mpiprocs', 'walltime']
}


landuse_categories_schema = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {
            'index': {'type': 'number'},
            'label': {'type': 'string'},
            'color': {'type': 'string'},
            'lcz': {'type': 'string'}
        },
        'required': ['index', 'color', 'label']
    }
}


landuse_map_schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'description': {'type': 'string'},
        'height': {'type': 'number'},
        'width': {'type': 'number'},
        'categories': landuse_categories_schema,
        'data': {'type': 'array'}
    },
    'required': ['name', 'height', 'width', 'categories', 'data']
}
