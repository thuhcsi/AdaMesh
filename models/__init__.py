def get_model(cfg):
    ## old
    if cfg.arch == 'stage1_vocaset':
        from models.stage1_vocaset import VQAutoEncoder as Model
        model = Model(args=cfg)
    elif cfg.arch == 'speech2landmark':
        from models.s2lmk import Speech2Landmark as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model