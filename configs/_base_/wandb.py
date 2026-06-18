# Weights & Biases logging via mmengine's native visualizer backend.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='georgiasouval',
            project='raw-detect',
            group='pascalraw',
            name=None,
            tags=['frozen-study'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
