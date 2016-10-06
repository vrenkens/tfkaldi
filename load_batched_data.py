


from prepare.prepareAurora import prepare_aurora
import prepare.prepare_timit.prepare_timit39


def load_batched_data(data_set):
    if data_set == 'aurora':
        print('loading_aurora_data')
        data_batches = prepare_aurora(BATCH_COUNT, BATCH_COUNT_VAL, BATCH_COUNT_TEST)
    else:
        data_batches = load_batched_timit39(BATCH_COUNT, BATCH_COUNT_VAL,
                                            BATCH_COUNT_TEST)
    return data_batches
    

    
